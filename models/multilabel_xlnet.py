import torch
import torch.nn as nn
from transformers import XLNetModel
from layers.attention import Attention
from layers.word_embeddings import WordRep


class MultilabelXLNet(nn.Module):
    def __init__(self, args=None, num_labels=None, dicts=None, label_ids=None):
        super(MultilabelXLNet, self).__init__()

        self.output_attentions = args.output_attentions
        
        if args.embed_labels:
            if label_ids is not None:
                self.embed_labels = True
                self.label_ids = torch.LongTensor(label_ids)
                if args.gpu:
                    self.label_ids = self.label_ids.cuda()
                self.word_rep = WordRep(args, dicts)
                self.label_proj = nn.Linear(self.label_ids.size(1), 1)
            else:
                print("embed_labels is set to True, but label_ids is None. Proceeding with randomly \
                       initialised label embeddings.")
                self.embed_labels = False
                self.label_ids = None
        else:
            self.embed_labels = False
            self.label_ids = None

        self.transformer = XLNetModel.from_pretrained(
            args.xlnet_base_model, dropout=args.transformer_dropout, summary_last_dropout=args.transformer_dropout,
            pad_token_id=1, bos_token_id=0, eos_token_id=2
        )
        
        self.final = nn.Linear(args.hidden_size, num_labels)
        nn.init.xavier_normal_(self.final.weight)
        
        if self.embed_labels:
            self.attn = Attention(
                q_dim=(num_labels, args.embed_size),
                k_dim=(None, args.hidden_size),
                learn_query=False,
                attn_type='general',
                normalise=False,
                dropout=args.attention_dropout
            )
        else:
            self.attn = Attention(
                q_dim=(num_labels, args.hidden_size),
                k_dim=(None, args.hidden_size),
                learn_query=True,
                attn_type='dot',
                normalise=False,
                dropout=args.attention_dropout
            )

        self.loss_function = nn.BCEWithLogitsLoss()
        
        if not args.update_transfomer_weights:
            self.freeze_transformer_params()
        
    def forward(self, input_ids, attention_mask, target=None):
        
        outputs = {}
        
        transformer_out = self.transformer(
            input_ids, attention_mask=attention_mask, model_output=self.output_attentions
        )
        
        last_hidden = transformer_out.last_hidden_state
        
        if self.embed_labels:
            label_attn_queries = self.label_proj(self.word_rep(self.label_ids).transpose(1, 2)).squeeze()
        else:
            label_attn_queries = None
        
        if self.output_attentions:
            self_attention = transformer_out.attentions 
            attn_out, label_attention = self.attn(
                last_hidden, queries=label_attn_queries, attention_mask=attention_mask, return_weights=True
            )
            outputs['self_attention'] = self_attention
            outputs['label_attention'] = label_attention            
        else:
            attn_out = self.attn(
                last_hidden, queries=label_attn_queries, attention_mask=attention_mask, return_weights=False
            )
        
        logits = self.final.weight.mul(attn_out).sum(dim=2).add(self.final.bias)
        
        if target is not None:
            loss = self.loss_function(logits, target)
            outputs['loss'] = loss
        
        outputs['output'] = torch.sigmoid(logits)
        
        return outputs
        
    def freeze_transformer_params(self):
        for p in self.transformer.parameters():
            p.requires_grad = False
