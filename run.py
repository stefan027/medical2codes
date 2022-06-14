import os
import pandas as pd
import numpy as np
import torch
import copy
import json
from transformers import BertTokenizerFast

from config.config import Config
from config.cli_arguments import read_cli_arguments
from data_utils.mlcdata import MLCData, MLCDataXLNet

from utils.utils import (
    get_doc_lengths,
    string_to_vec,
    f1_score,
    expand_labels,
    get_ancestors,
    replace_unknown,
    caml_tokenize,
    labels_desc_to_ids,
    get_label_counts,
    get_label_index_mappings,
    build_vocab
)
from utils.read_model_config_file import read_args_from_csv
from training.train_test import pytorch_training_loop, pytorch_testing_loop
from models.initialise_model import initialise_model

cli_args = read_cli_arguments()

if cli_args.random_seed is not None:
    torch.manual_seed(cli_args.random_seed)
    np.random.seed(cli_args.random_seed)

# Create output directory if it doesn't exist
if not os.path.exists(cli_args.output_dir):
    os.makedirs(cli_args.output_dir)

if cli_args.mimic_version == 2:
    df_train = pd.read_csv(os.path.join(cli_args.data_dir, 'mimic2_train.txt'), sep='|')
    df_val = pd.read_csv(os.path.join(cli_args.data_dir, 'mimic2_valid.txt'), sep='|')
    df_test = pd.read_csv(os.path.join(cli_args.data_dir, 'mimic2_test.txt'), sep='|')
else:
    df_train = pd.read_csv(os.path.join(cli_args.data_dir, 'mimic3_train.csv'), sep='|')
    df_val = pd.read_csv(os.path.join(cli_args.data_dir, 'mimic3_valid.csv'), sep='|')
    df_test = pd.read_csv(os.path.join(cli_args.data_dir, 'mimic3_test.csv'), sep='|')

if cli_args.vocab_file:
    vocab = []
    with open(cli_args.vocab_file, 'r') as f:
        lines = f.readlines()
        for token in lines:
            vocab.append(token.strip('\n'))
    vocab = set(vocab)
    df_train['TEXT'] = df_train['TEXT'].apply(lambda row: replace_unknown(row, vocab))
    df_val['TEXT'] = df_val['TEXT'].apply(lambda row: replace_unknown(row, vocab))
    df_test['TEXT'] = df_test['TEXT'].apply(lambda row: replace_unknown(row, vocab))

if cli_args.train_expanded_labels:
    icd_parent_child_relations = pd.read_csv(
        os.path.join(cli_args.data_dir, 'ICD9_parent_child_relations_updated'),
        sep='\t', header=None, names=['Parent', 'Child']
    )
    parents = icd_parent_child_relations.Parent.to_list()
    children = icd_parent_child_relations.Child.to_list()
    child_to_parent = {child: parent for child, parent in zip(children, parents)}

    df_train['Label'] = expand_labels(df_train['Label'].to_list(), child_to_parent)
    df_val['Label'] = expand_labels(df_val['Label'].to_list(), child_to_parent)
    df_test['Label'] = expand_labels(df_test['Label'].to_list(), child_to_parent)

max_codes = cli_args.max_codes   # or None

# Get label counts
df = pd.concat([df_train, df_val, df_test])
label_counts, raw_labels = get_label_counts(
    raw_labels=df['Label'].to_list(),
    max_codes=max_codes
)

# Get label to index cross-mappings
label_to_ix, ix_to_label = get_label_index_mappings(raw_labels)
num_labels = len(label_to_ix)
print(f"Number of labels: {num_labels}")

# Build vocabulary
df = pd.concat([df_train, df_val])
vocab, vocab_lookup, vocab_reverse_lookup = build_vocab(df['TEXT'])
print(f"Vocabulary size: {len(vocab)}")

if cli_args.config.split('.')[-1] == 'csv':
    config_dict = read_args_from_csv(cli_args.config)
elif cli_args.config.split('.')[-1] == 'json':
    with open(cli_args.config, 'r') as f:
        config_dict = json.load(f)
else:
    raise Exception("Supported config file types: csv and json")

model_names = config_dict.keys()

with open(os.path.join(cli_args.output_dir, 'model_accuracy.csv'), 'w') as f:
    f.write('model_name,val_f1,test_f1\n')

for model_name in model_names:

    args = Config(config_dict[model_name])
    args.model_weights_load_path = cli_args.model_path
    
    print(f'Training {model_name}')
    print(args.__dict__)

    device = 'cuda' if args.gpu else 'cpu'
    
    args.model_weights_save_path = os.path.join(cli_args.output_dir, f'{model_name}.pt')
    
    if args.model_type == 'XLNet':
        dicts = {
            'ind2w': {},
            'w2ind': {},
            'ind2c': ix_to_label,
            'c2ind': label_to_ix
        }
    else:
        dicts = {
            'ind2w': vocab_reverse_lookup,
            'w2ind': vocab_lookup,
            'ind2c': ix_to_label,
            'c2ind': label_to_ix
        }

    if args.embed_labels:
        label_ids, dicts = labels_desc_to_ids(os.path.join(cli_args.data_dir, 'ICD9_descriptions_updated'), dicts)
    else:
        label_ids = None
    
    if args.model_type == 'XLNet':
        args.xlnet_base_model = cli_args.xlnet_base_model
        tokenizer = BertTokenizerFast.from_pretrained('mimic_wordpiece')
        tokenizer.model_max_length = 3072
        tokenizer.init_kwargs['model_max_length'] = 3072
        train_data = MLCDataXLNet(df_train, dicts, tokenizer, device)
        valid_data = MLCDataXLNet(df_val, dicts, tokenizer, device)
        test_data = MLCDataXLNet(df_test, dicts, tokenizer, device)
    else:
        train_data = MLCData(df_train, dicts, args.max_doc_length, device)
        valid_data = MLCData(df_val, dicts, args.max_doc_length, device)
        test_data = MLCData(df_test, dicts, args.max_doc_length, device)
        
    if not cli_args.eval_only:    
        
        model = initialise_model(args, num_labels, dicts, label_ids=label_ids)
        model = model.to(device)
    
        if args.model_weights_load_path:
            model.load_state_dict(torch.load(args.model_weights_load_path))
            print(f'Model weights loaded from {args.model_weights_load_path}')
        
        if args.model_type == 'XLNet':
            if args.embed_labels:
                optimizer = torch.optim.Adam([
                    {'params': model.transformer.parameters(), 'lr': args.lm_learning_rate},
                    {'params': model.attn.parameters()},
                    {'params': model.word_rep.parameters()},
                    {'params': model.label_proj.parameters()},
                    {'params': model.final.parameters()}
                    ], lr=args.learning_rate)
            else:
                optimizer = torch.optim.Adam([
                    {'params': model.transformer.parameters(), 'lr': args.lm_learning_rate},
                    {'params': model.attn.parameters()},
                    {'params': model.final.parameters()}
                    ], lr=args.learning_rate)
        else:
            optimizer = torch.optim.Adam(model.parameters(), weight_decay=0, lr=args.learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1, factor=0.1)
        
        val_score = pytorch_training_loop(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_data=train_data,
            val_data=valid_data,
            num_labels=num_labels,
            start_epoch=args.start_epoch,
            total_epochs=args.epochs,
            batch_size=args.batch_size,
            model_save_path=args.model_weights_save_path,
            best_val_score=args.best_val_score,
            patience=args.patience,
            shuffle_data=cli_args.shuffle
        )
        args.model_weights_load_path = args.model_weights_save_path
    
    # Test
    model = initialise_model(args, num_labels, dicts, label_ids=label_ids)

    model = model.to(device)

    if args.model_weights_load_path:
        model.load_state_dict(torch.load(args.model_weights_load_path))
    
    if cli_args.eval_data == "train":
        data = train_data
    elif cli_args.eval_data == "val":
        data = valid_data
    else:
        data = test_data
        
    n_test = data.num_examples

    test_probs_all, test_targs_all = pytorch_testing_loop(
        model=model,
        test_data=data,
        num_labels=num_labels,
        batch_size=args.batch_size
    )

    p, r, f1 = f1_score(test_probs_all, test_targs_all, 0.5, average='micro')
    print('Precision ({}): {:.4f}, Recall ({}): {:.4f}, F1 ({}): {:.4f}'.format(
        cli_args.eval_data, p, cli_args.eval_data, r, cli_args.eval_data, f1)
    )

    output_file = os.path.join(cli_args.output_dir, f"{model_name}_{cli_args.eval_data}_preds.csv")
    hier_output_file = os.path.join(cli_args.output_dir, f"{model_name}_{cli_args.eval_data}_hier_preds.csv")

    # Create output file
    with open(output_file, 'w') as f:
        f.write('patient_id,predicted_codes\n')
        for i in range(n_test):
            outstr = str(data.df.patient_id[i])
            outstr += ',"'
            for j in range(len(label_to_ix)):
                if test_probs_all[i, j] > 0.5:
                    outstr += ix_to_label[j]
                    outstr += ' '
            outstr = outstr.strip()
            outstr += '"\n'
            f.write(outstr)
    
    if not cli_args.eval_only:
        with open(os.path.join(cli_args.output_dir, 'model_accuracy.csv'), 'a') as f:
            f.write('{},{:.4f},{:.4f}\n'.format(model_name, args.best_val_score, f1))

    if cli_args.train_expanded_labels:
        # Create hierarchical output file
        hier_preds = copy.deepcopy(test_probs_all)
        hier_preds = (hier_preds >= 0.5).astype(np.int32)

        for i in range(len(label_to_ix)):
            label = ix_to_label[i]
            ancestors = get_ancestors(label, child_to_parent)
    
            if len(ancestors) == 1:
                continue
    
            label_ix = label_to_ix[label]

            for parent in ancestors:
                parent_ix = label_to_ix[parent]
                hier_preds[:, label_ix] *= hier_preds[:, parent_ix]

        with open(hier_output_file, 'w') as f:
            f.write('patient_id,predicted_codes\n')
            for i in range(n_test):
                outstr = str(data.df.patient_id[i])
                outstr += ',"'
                for j in range(len(label_to_ix)):
                    if hier_preds[i, j] == 1:
                        outstr += ix_to_label[j]
                        outstr += ' '
                outstr = outstr.strip()
                outstr += '"\n'
                f.write(outstr)
