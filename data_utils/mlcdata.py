import numpy as np
import torch
from utils.utils import remove_tokens, get_doc_lengths, icd_string_to_vec, string_to_vec


class MLCData:
    def __init__(self, df, dicts, max_doc_length, device):
        self.df = df
        self.dicts = dicts
        self.max_doc_length = max_doc_length
        self.device = device
        self.num_examples = len(df)

    def get_batch(self, s_idx, e_idx, return_labels=True):
        input_dict = self.get_batch_text(s_idx, e_idx)
        if return_labels:
            input_dict['target'] = self.get_batch_labels(s_idx, e_idx)
        return input_dict
        
    def get_batch_text(self, s_idx, e_idx):
        batch_documents = self.df.TEXT.to_list()[s_idx:e_idx]
        batch_documents = remove_tokens(batch_documents, ['[CLS]', '[SEP]'])
        batch_lengths = get_doc_lengths(batch_documents)
        max_batch_length = min(self.max_doc_length, max(batch_lengths))
        inputs, batch_lengths = string_to_vec(batch_documents, max_batch_length, self.dicts['w2ind'])
        data = torch.LongTensor(inputs).to(self.device)
        return {'x': data, 'lengths': batch_lengths}
    
    def get_batch_labels(self, s_idx, e_idx):
        batch_labels = self.df.Label.to_list()[s_idx:e_idx]
        targets = icd_string_to_vec(batch_labels, len(self.dicts['c2ind']), self.dicts['c2ind'])
        return torch.FloatTensor(targets).to(self.device)

    def shuffle(self, random_seed=None):
        self.df = self.df.sample(frac=1, replace=False, random_state=random_seed)


class MLCDataXLNet:
    def __init__(self, df, dicts, tokenizer, device):
        self.df = df
        self.dicts = dicts
        self.tokenizer = tokenizer
        self.device = device
        self.num_examples = len(df)
        
    def get_batch(self, s_idx, e_idx, return_labels=True):
        input_dict = self.get_batch_text(s_idx, e_idx)
        if return_labels:
            input_dict['target'] = self.get_batch_labels(s_idx, e_idx)
        return input_dict
        
    def get_batch_text(self, s_idx, e_idx):
        batch_documents = self.df.TEXT.to_list()[s_idx:e_idx]
        batch_documents = remove_tokens(batch_documents, ['[CLS]', '[SEP]'])
        input_dict = self.tokenizer.batch_encode_plus(batch_documents, truncation=True, padding=True)
        input_ids = torch.LongTensor(input_dict['input_ids']).to(self.device)
        attention_mask = torch.LongTensor(input_dict['attention_mask']).to(self.device)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}
    
    def get_batch_labels(self, s_idx, e_idx):
        batch_labels = self.df.Label.to_list()[s_idx:e_idx]
        targets = icd_string_to_vec(batch_labels, len(self.dicts['c2ind']), self.dicts['c2ind'])
        return torch.FloatTensor(targets).to(self.device)
    
    def shuffle(self, random_seed=None):
        self.df = self.df.sample(frac=1, replace=False, random_state=random_seed)


class MLCDataBLUE:
    def __init__(self, df, label2idx, tokenizer, device):
        self.df = df
        self.label2idx = label2idx
        self.preprocess_text()
        self.tokenizer = tokenizer
        self.device = device
        self.num_examples = len(df)

    def preprocess_text(self):
        self.df['sentence_original'] = self.df.sentence
        self.df['sentence'] = self.df.sentence.apply(lambda x: x.lower())

    def get_batch(self, s_idx, e_idx, return_labels=True):
        input_dict = self.get_batch_text(s_idx, e_idx)
        if return_labels:
            input_dict['target'] = self.get_batch_labels(s_idx, e_idx)
        return input_dict

    def get_batch_text(self, s_idx, e_idx):
        batch_documents = self.df.sentence.to_list()[s_idx:e_idx]
        input_dict = self.tokenizer.batch_encode_plus(batch_documents, truncation=True, padding=True)
        input_ids = torch.LongTensor(input_dict['input_ids']).to(self.device)
        attention_mask = torch.LongTensor(input_dict['attention_mask']).to(self.device)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    def get_batch_labels(self, s_idx, e_idx):
        return self.label2vec(self.df.labels.to_list()[s_idx:e_idx])

    def shuffle(self, random_seed=None):
        self.df = self.df.sample(frac=1, replace=False, random_state=random_seed)

    def label2vec(self, labels):
        lbl_vec = np.zeros((len(labels), len(self.label2idx)))

        for i, label_str in enumerate(labels):
            if type(label_str) == str:
                for lbl in label_str.split(','):
                    lbl_idx = self.label2idx[lbl] - 1
                    lbl_vec[i, lbl_idx] = 1
        return torch.FloatTensor(lbl_vec).to(self.device)
