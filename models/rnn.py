import torch
import torch.nn as nn
import torch.nn.functional as F
             
from layers.word_embeddings import WordRep
from layers.attention import Attention


class EncoderRNN(nn.Module):
    def __init__(self, args, dicts, max_len, embedding_size, hidden_size, dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='gru', variable_lengths=False):
        super(EncoderRNN, self).__init__()

        self.variable_lengths = variable_lengths
        self.bidirectional = bidirectional

        self.word_rep = WordRep(args, dicts)

        if rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        elif rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM

        self.rnn = self.rnn_cell(
            embedding_size, hidden_size, n_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout_p
        )

    def forward(self, input_var, input_lengths=None):
        embedded = self.word_rep(input_var)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, input_lengths, batch_first=True, enforce_sorted=False
            )
        output, hidden = self.rnn(embedded)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden


class RNNTextClassifier(nn.Module):
    def __init__(self, args, num_labels, dicts, label_ids):
        super(RNNTextClassifier, self).__init__()

        self.attention = args.attention
        self.attention_type = args.attention_type
        self.embed_labels = args.embed_labels

        if args.bidirectional:
            output_size = args.hidden_size * 2
        else:
            output_size = args.hidden_size

        if label_ids is not None:
            self.label_ids = torch.LongTensor(label_ids)
            if args.gpu:
                self.label_ids = self.label_ids.cuda()
        else:
            self.label_ids = None

        # Label embeddings
        if self.embed_labels:
            self.label_ids = torch.LongTensor(label_ids)
            if args.gpu:
                self.label_ids = self.label_ids.cuda()

            self.label_proj = nn.Linear(self.label_ids.size(1), 1)
        else:
            self.label_ids = None

        self.encoder = EncoderRNN(
            args,
            dicts,
            rnn_cell=args.rnn_cell_type,
            dropout_p=args.rnn_dropout,
            embedding_size=args.embed_size,
            max_len=args.max_doc_length,
            hidden_size=args.hidden_size,
            n_layers=args.num_layers,
            variable_lengths=True,
            bidirectional=args.bidirectional
        )

        if self.attention and self.attention_type in ('labelwise', 'labelembed'):
            self.predictor = nn.Linear(output_size, num_labels)
        else:
            self.predictor = nn.Linear(output_size*3, num_labels)

        if self.attention and self.attention_type == 'dot':
            self.attn = Attention(
                q_dim=(None, output_size),
                k_dim=(1, output_size),
                learn_query=False,
                attn_type='dot',
                normalise=False,
                dropout=args.attention_dropout
            )
        elif self.attention and self.attention_type == 'general':
            self.attn = Attention(
                q_dim=(None, output_size),
                k_dim=(1, output_size),
                learn_query=False,
                attn_type='general',
                normalise=False,
                dropout=args.attention_dropout
            )
        elif self.attention and self.attention_type == 'labelwise':
            self.attn = Attention(
                q_dim=(num_labels, output_size),
                k_dim=(None, output_size),
                learn_query=True,
                normalise=False,
                dropout=args.attention_dropout
            )
        elif self.attention and self.attention_type == 'labelembed':
            self.attn = Attention(
                q_dim=(num_labels, args.embed_size),
                k_dim=(None, output_size),
                learn_query=False,
                attn_type='general',
                normalise=False,
                dropout=args.attention_dropout
            )

        self.loss_function = nn.BCEWithLogitsLoss()
        
    def forward(self, x, target, text_inputs=None, desc_data=None, lengths=None):
        output, _ = self.encoder(x, lengths)  # output.shape = (batch_size, num_seq, hidden_size)

        idx = (torch.LongTensor(lengths) - 1).view(-1, 1).expand(
            len(lengths), output.size(2))
        time_dimension = 1
        idx = idx.unsqueeze(time_dimension)
        if output.is_cuda:
            idx = idx.cuda(output.data.get_device())
        last_output = output.gather(
            time_dimension, torch.autograd.Variable(idx)).squeeze(time_dimension)
        maxpool_output = output.permute(0, 2, 1)
        maxpool_output = F.max_pool1d(maxpool_output, maxpool_output.size(2)).squeeze(2)
        avgpool_output = output.permute(0, 2, 1)
        avgpool_output = F.avg_pool1d(avgpool_output, avgpool_output.size(2)).squeeze(2)
        if not self.attention:
            out = torch.cat([last_output, maxpool_output, avgpool_output], 1)
            pred = self.predictor(out).squeeze()
        else:
            if self.attention_type in ('dot', 'general'):
                attn_out, attn_weights = self.attn(output, last_output.unsqueeze(1), return_weights=True)
                out = torch.cat([attn_out, maxpool_output, avgpool_output], 1)
                pred = self.predictor(out).squeeze()
            elif self.attention_type == 'labelwise':
                attn_out = self.attn(output)
                pred = self.predictor.weight.mul(attn_out).sum(dim=2).add(self.predictor.bias)
            elif self.attention_type == 'labelembed':
                label_attn_queries = self.label_proj(self.encoder.word_rep(self.label_ids).transpose(1,2)).squeeze()
                attn_out = self.attn(output, label_attn_queries)
                pred = self.predictor.weight.mul(attn_out).sum(dim=2).add(self.predictor.bias)
            else:
                raise Exception('attention type: %s is not supported' % self.attention_type)

        loss = self.loss_function(pred, target)
        y = torch.sigmoid(pred)
        
        return {'output': y, 'loss': loss}
