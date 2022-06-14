import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_ as xavier_uniform
from math import floor

from layers.word_embeddings import WordRep
from layers.attention import Attention


class MultiCNN(nn.Module):
    def __init__(self, args, num_labels, dicts, label_ids=None):
        super(MultiCNN, self).__init__()

        self.word_rep = WordRep(args, dicts)
        
        self.use_attention = args.attention
        
        if args.embed_labels:
            if label_ids is not None:
                self.embed_labels = True
                self.label_ids = torch.LongTensor(label_ids)
                if args.gpu:
                    self.label_ids = self.label_ids.cuda()
                self.label_proj1 = nn.Linear(self.label_ids.size(1), 1)
                self.label_proj2 = nn.Linear(self.label_ids.size(1), 1)
            else:
                print("embed_labels is set to True, but label_ids is None. Proceeding with randomly \
                       initialised label embeddings.")
                self.embed_labels = False
                self.label_ids = None
        else:
            self.embed_labels = False
            self.label_ids = None

        if type(args.filter_size) == int:
            self.filter_num = 1
            filter_size = int(args.filter_size)
            self.conv = nn.Conv1d(
                self.word_rep.feature_size, args.num_filter_maps, kernel_size=filter_size,
                padding=int(floor(filter_size / 2))
            )
            xavier_uniform(self.conv.weight)
        else:
            filter_sizes = args.filter_size.split(',')
            self.filter_num = len(filter_sizes)
            self.conv = nn.ModuleList()
            for filter_size in filter_sizes:
                filter_size = int(filter_size)
                tmp = nn.Conv1d(
                    self.word_rep.feature_size, args.num_filter_maps, kernel_size=filter_size,
                    padding=int(floor(filter_size / 2))
                )
                xavier_uniform(tmp.weight)
                self.conv.add_module('conv-{}'.format(filter_size), tmp)

        if self.use_attention:
            if self.embed_labels:
                self.attn = Attention(
                    q_dim=(num_labels, args.embed_size),
                    k_dim=(None, self.filter_num*args.num_filter_maps),
                    learn_query=False,
                    attn_type='general',
                    normalise=False,
                    dropout=args.attention_dropout
                )
            else:
                self.attn = Attention(
                    q_dim=(num_labels, self.filter_num*args.num_filter_maps),
                    k_dim=(None, self.filter_num*args.num_filter_maps),
                    learn_query=True,
                    attn_type='dot',
                    normalise=False,
                    dropout=args.attention_dropout
                )
                
        self.final = nn.Linear(self.filter_num*args.num_filter_maps, num_labels)
        nn.init.xavier_normal_(self.final.weight)
        
        self.loss_function = nn.BCEWithLogitsLoss()
        
    def forward(self, x, target, lengths=None):

        x = self.word_rep(x)

        x = x.transpose(1, 2)

        if self.filter_num == 1:
            x = torch.tanh(self.conv(x).transpose(1, 2))
        else:
            conv_result = []
            for tmp in self.conv:
                conv_result.append(torch.tanh(tmp(x).transpose(1, 2)))
            x = torch.cat(conv_result, dim=2)

        # Embed labels
        if self.embed_labels:
            label_attn_queries = self.label_proj(self.word_rep(self.label_ids).transpose(1, 2)).squeeze()
        else:
            label_attn_queries = None
        
        if self.use_attention:
            attn_out = self.attn(x, queries=label_attn_queries)
            logits = self.final.weight.mul(attn_out).sum(dim=2).add(self.final.bias)
        else:
            m = torch.max(x.transpose(1, 2), dim=2)[0]
            logits = self.final(m)
        
        loss = self.loss_function(logits, target)
        y = torch.sigmoid(logits)
        
        return {'output': y, 'loss': loss}


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, use_res, dropout):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(
                inchannel, outchannel, kernel_size=kernel_size, stride=stride,
                padding=int(floor(kernel_size / 2)), bias=False
            ),
            nn.BatchNorm1d(outchannel),
            nn.Tanh(),
            nn.Conv1d(
                outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=int(floor(kernel_size / 2)),
                bias=False
            ),
            nn.BatchNorm1d(outchannel)
        )

        self.use_res = use_res
        if self.use_res:
            self.shortcut = nn.Sequential(
                nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(outchannel)
            )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.left(x)
        if self.use_res:
            out += self.shortcut(x)
        out = torch.tanh(out)
        out = self.dropout(out)
        return out


class MultiResCNN(nn.Module):

    def __init__(self, args, num_labels, dicts, label_ids=None):
        super(MultiResCNN, self).__init__()

        self.word_rep = WordRep(args, dicts)

        self.use_attention = args.attention
        
        if args.embed_labels:
            if label_ids is not None:
                self.embed_labels = True
                self.label_ids = torch.LongTensor(label_ids)
                if args.gpu:
                    self.label_ids = self.label_ids.cuda()
                self.label_proj = nn.Linear(self.label_ids.size(1), 1)
            else:
                print("embed_labels is set to True, but label_ids is None. Proceeding with randomly \
                       initialised label embeddings.")
                self.embed_labels = False
                self.label_ids = None
        else:
            self.embed_labels = False
            self.label_ids = None

        self.conv = nn.ModuleList()

        if type(args.filter_size) == int:
            self.filter_num = 1

            conv_dimension = self.word_rep.conv_dict[args.conv_layer]
            for idx in range(args.conv_layer):
                tmp = ResidualBlock(
                    conv_dimension[idx], conv_dimension[idx + 1], int(args.filter_size), 1, True, args.residual_dropout
                )
                self.conv.add_module('conv-{}'.format(idx), tmp)

        else:
            filter_sizes = args.filter_size.split(',')

            self.filter_num = len(filter_sizes)
            for filter_size in filter_sizes:
                filter_size = int(filter_size)
                one_channel = nn.ModuleList()
                tmp = nn.Conv1d(
                    self.word_rep.feature_size, self.word_rep.feature_size, kernel_size=filter_size,
                    padding=int(floor(filter_size / 2))
                )
                xavier_uniform(tmp.weight)
                one_channel.add_module('baseconv', tmp)

                conv_dimension = self.word_rep.conv_dict[args.conv_layer]
                for idx in range(args.conv_layer):
                    tmp = ResidualBlock(
                        conv_dimension[idx], conv_dimension[idx + 1], filter_size, 1, True, args.residual_dropout
                    )
                    one_channel.add_module('resconv-{}'.format(idx), tmp)

                self.conv.add_module('channel-{}'.format(filter_size), one_channel)

        if self.use_attention:
            if self.embed_labels:
                self.attn = Attention(
                    q_dim=(num_labels, args.embed_size),
                    k_dim=(None, self.filter_num*args.num_filter_maps),
                    learn_query=False,
                    attn_type='general',
                    normalise=False,
                    dropout=args.attention_dropout
                )
            else:
                self.attn = Attention(
                    q_dim=(num_labels, self.filter_num*args.num_filter_maps),
                    k_dim=(None, self.filter_num*args.num_filter_maps),
                    learn_query=True,
                    attn_type='dot',
                    normalise=False,
                    dropout=args.attention_dropout
                )
        
        self.final = nn.Linear(self.filter_num*args.num_filter_maps, num_labels)
        nn.init.xavier_normal_(self.final.weight)
        
        self.loss_function = nn.BCEWithLogitsLoss()
        
    def forward(self, x, target, lengths=None):

        x = self.word_rep(x)

        x = x.transpose(1, 2)

        conv_result = []
        for conv in self.conv:
            tmp = x
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.tanh(md(tmp))
                else:
                    tmp = md(tmp)
            tmp = tmp.transpose(1, 2)
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2)

        # Embed labels
        if self.embed_labels:
            label_attn_queries = self.label_proj(self.word_rep(self.label_ids).transpose(1, 2)).squeeze()
        else:
            label_attn_queries = None
        
        if self.use_attention:
            attn_out = self.attn(x, queries=label_attn_queries)
            logits = self.final.weight.mul(attn_out).sum(dim=2).add(self.final.bias)
        else:
            m = torch.max(x.transpose(1, 2), dim=2)[0]
            logits = self.final(m)
            
        loss = self.loss_function(logits, target)
        y = torch.sigmoid(logits)
        
        return {'output': y, 'loss': loss}
