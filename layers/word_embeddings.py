# Adapted from Li et al. (2019)

import torch
import torch.nn as nn
import numpy as np
from gensim.models import KeyedVectors


class WordRep(nn.Module):
    def __init__(self, args, dicts):
        super(WordRep, self).__init__()

        self.gpu = args.gpu

        init_width = 0.5 / args.embed_size

        if args.embed_file:
            print("Loading word embeddings from {}".format(args.embed_file))
            embeddings_np = np.random.uniform(-0.25, 0.25, (len(dicts['w2ind']), args.embed_size))
            model = KeyedVectors.load_word2vec_format(args.embed_file, binary=True)
            unmapped_tokens = 0
            for k, v in dicts['w2ind'].items():
                if k in vars(model)['key_to_index']:
                    curr_embedding = model.get_vector(k)
                    embeddings_np[v] = curr_embedding
                else:
                    unmapped_tokens += 1

            print(f"Total dictionary size: {len(dicts['w2ind'])}")
            print(f"Pre-trained embeddings loaded for {len(dicts['w2ind'])-unmapped_tokens} tokens")
            print(f"{unmapped_tokens} tokens randomly initialised")

            w = torch.from_numpy(embeddings_np).float()
            self.embed = nn.Embedding(w.size()[0], w.size()[1], padding_idx=0)
            self.embed.weight.data = w.clone()
        else:
            self.embed = nn.Embedding(len(dicts['w2ind']), args.embed_size, padding_idx=dicts['w2ind']['<pad>'])
            torch.nn.init.uniform_(self.embed.weight, a=-init_width, b=init_width)

        self.feature_size = self.embed.embedding_dim

        self.embed_drop = nn.Dropout(p=args.embedding_dropout)

        if args.model_type in ('CNN', 'ResCNN'):
            self.conv_dict = {
                1: [self.feature_size, args.num_filter_maps],
                2: [self.feature_size, 100, args.num_filter_maps],
                3: [self.feature_size, 150, 100, args.num_filter_maps],
                4: [self.feature_size, 200, 150, 100, args.num_filter_maps]
            }
        
        if not args.update_word_embeddings:
            self.freeze_net()

    def forward(self, x):

        features = [self.embed(x)]

        x = torch.cat(features, dim=2)

        x = self.embed_drop(x)
        return x

    def freeze_net(self):
        for p in self.embed.parameters():
            p.requires_grad = False
