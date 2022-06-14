import torch
import torch.nn as nn
import torch.nn.functional as f


class Attention(nn.Module):
    def __init__(self, q_dim, k_dim, learn_query, attn_type='dot', normalise=False, dropout=0.1):
        super(Attention, self).__init__()
        
        self.learn_query = learn_query
        self.normalise = normalise
        self.attn_type = attn_type
        
        if self.attn_type == 'general':
            self.W = nn.Linear(k_dim[-1], q_dim[-1]).weight
            nn.init.xavier_normal_(self.W)
            
        if self.learn_query:
            self.queries = nn.Linear(q_dim[-1], q_dim[-2]).weight
            nn.init.xavier_normal_(self.queries)
        
        if self.normalise:
            self.norm = nn.LayerNorm(k_dim[-1])
            
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, keys, queries=None, return_weights=False, attention_mask=None):
        
        if self.learn_query:
            queries = self.queries
        
        # Compute attention
        if attention_mask is not None:
            label_attentions = []
            m = torch.zeros(keys.size(0), queries.size(0), keys.size(2), device=keys.device)
            for b in range(keys.size(0)):
                seq_len = sum(attention_mask[b, :])
                max_len = attention_mask.size(1)
                keys_ = keys[b, (max_len-seq_len+1):-1, :].unsqueeze(0)
                
                if self.attn_type == 'dot':
                    alpha = f.softmax(queries.matmul(keys_.transpose(1, 2)), dim=2)
                elif self.attn_type == 'general':
                    alpha = f.softmax(queries.matmul(self.W).matmul(keys_.transpose(1, 2)), dim=2)
                else:
                    raise NotImplementedError("Error: attn_type must be dot or general")
                    
                m[b, :, :] = alpha.matmul(keys_)
                label_attentions.append(alpha.detach())
        else:
            if self.attn_type == 'dot':
                align = queries.matmul(keys.transpose(1, 2))
            elif self.attn_type == 'general':
                align = queries.matmul(self.W).matmul(keys.transpose(1, 2))
            else:
                raise NotImplementedError("Error: attn_type must be dot or general")
            alpha = f.softmax(align, dim=2)
            m = alpha.matmul(keys)
            label_attentions = [alpha[b, :, :].detach() for b in range(alpha.size(0))]
            
        if m.size(1) == 1:
            m = m.squeeze(1)

        if self.normalise:
            m = self.norm(m)
       
        # Drop-out
        out = self.dropout(m)
        
        if return_weights:
            return out, label_attentions
        else:
            return out
