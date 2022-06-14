# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 16:34:16 2021

@author: stefa
"""

import pandas as pd
import numpy as np
import torch
from data_utils.mlcdata import MLCDataBLUE
from training.train_test import pytorch_training_loop, pytorch_testing_loop
from models.multilabel_xlnet import MultilabelXLNet
from transformers import BertTokenizerFast
from utils.utils_blue import combine_abstract

torch.manual_seed(0)
np.random.seed(0)

device = 'cpu'

df_train = pd.read_csv('../../../../BLUE/data/hoc/train.tsv', sep='\t')
df_valid = pd.read_csv('../../../../BLUE/data/hoc/dev.tsv', sep='\t')
df_test = pd.read_csv('../../../../BLUE/data/hoc/test.tsv', sep='\t')

labels = df_train.labels.to_list()
label_freq = {}
for lbl_str in labels:
    if type(lbl_str) == str:
        for lbl in lbl_str.split(','):
            if lbl in label_freq:
                label_freq[lbl] += 1
            else:
                label_freq[lbl] = 1
print("Number of labels: {}".format(len(label_freq)))

label2idx = {lbl: i+1 for i, lbl in enumerate(label_freq.keys())}

df = pd.concat([df_train, df_valid, df_test])
sent_idx = df['index'].to_list()
abst_idx = set([idx.split('_')[0] for idx in sent_idx])
print("Number of abstracts: {}".format(len(abst_idx)))


#df_train = remove_sent_wo_labels(df_train)
#df_valid = remove_sent_wo_labels(df_valid)
#df_test = remove_sent_wo_labels(df_test)

tokenizer = BertTokenizerFast.from_pretrained('../../../Baselines/wordpiece_tokenizer/mimic-wordpiece')
tokenizer.model_max_length = 3072
tokenizer.init_kwargs['model_max_length'] = 3072
        
#train_data = MLCData(df_train, label2idx, tokenizer, device)
#valid_data = MLCData(df_valid, label2idx, tokenizer, device)
#test_data = MLCData(df_test, label2idx, tokenizer, device)

train_data = MLCDataBLUE(combine_abstract(df_train), label2idx, tokenizer, device)
valid_data = MLCDataBLUE(combine_abstract(df_valid), label2idx, tokenizer, device)
test_data = MLCDataBLUE(combine_abstract(df_test), label2idx, tokenizer, device)

class Arguments:
    def __init__(self):
        self.hidden_size = 256
        self.transformer_dropout = 0.1
        self.attention_dropout = 0.1
        self.update_transfomer_weights = True
        self.output_attentions = False
        self.embed_labels = False
        
args = Arguments()
        
model = MultilabelXLNet(args=args, Y=10)
"""
optimizer = torch.optim.Adam([
    {'params': model.transformer.parameters(), 'lr': 2e-05},
    {'params': model.attn.parameters()},
    {'params': model.final.parameters()}
    ], lr=3e-04)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1, factor=0.9)
        
pytorch_training_loop(model, optimizer, scheduler,
                          train_data, valid_data, 10,
                          0, 50, 4,
                          'mlc_model.pt',
                          best_val_score = 0,
                          #gpu = False,
                          patience = 5,
                          shuffle_data=True)
"""
model.load_state_dict(torch.load('../blue_mlc_model.pt'))
test_probs, test_targs = pytorch_testing_loop(
    model, test_data, 10, 4
)

#from utils import f1_score
#print(f1_score(test_probs, test_targs, 0.5, average='micro'))

"""
idx2label = {v: k for k, v in label2idx.items()}
preds = {}
labels = {}
for i, sent_idx in enumerate(df_test['index'].to_list()):
    abst_idx = sent_idx.split('_')[0]
    if abst_idx not in preds:
        preds[abst_idx] = set()
        labels[abst_idx] = set()
    #for j in range(1, test_probs.shape[1], 1):
    for j in range(test_probs.shape[1]):
        if test_probs[i, j] >= 0.5:
            preds[abst_idx].add(idx2label[j+1])
        if test_targs[i, j] >= 0.5:
            labels[abst_idx].add(idx2label[j+1])
"""
idx2label = {v: k for k, v in label2idx.items()}
preds = {}
labels = {}
i = 0
for sent_idx in df_test['index'].to_list():
    abst_idx = sent_idx.split('_')[0]
    if abst_idx not in preds:
        preds[abst_idx] = set()
        labels[abst_idx] = set()
        for j in range(test_probs.shape[1]):
            if test_probs[i, j] >= 0.5:
                preds[abst_idx].add(idx2label[j+1])
            if test_targs[i, j] >= 0.5:
                labels[abst_idx].add(idx2label[j+1])
        i += 1
            

true_positives = 0
ground_truth_positives = 0
predicted_positives = 0

for idx in preds.keys():
    P = set(preds[idx])
    T = set(labels[idx])
    true_positives += len(P.intersection(T))
    ground_truth_positives += len(T)
    predicted_positives += len(P)

p = true_positives/predicted_positives
r = true_positives/ground_truth_positives
f = (2*p*r) / (p+r)
print(p, r, f)

# Example-based metrics
calc_precision = 0
calc_recall = 0

for idx in preds.keys():
    P = set(preds[idx])
    T = set(labels[idx])
    if len(P) > 0:
        calc_precision += len(P.intersection(T))/len(P)
    if len(T) > 0:
        calc_recall += len(P.intersection(T))/len(T)

p = calc_precision/len(preds)
r = calc_recall/len(preds)
f = (2*p*r) / (p+r)
print(p, r, f)