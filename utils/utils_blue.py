import pandas as pd


def combine_abstract(df):
    sentence_dict = {}
    labels_dict = {}

    for idx, sent, lbl in zip(df['index'].to_list(), df['sentence'].to_list(), df['labels'].to_list()):
        abst_idx = idx.split('_')[0]

        if abst_idx in sentence_dict:
            sentence_dict[abst_idx] += '\n{}'.format(sent)
        else:
            sentence_dict[abst_idx] = sent
            labels_dict[abst_idx] = set()

        if type(lbl) == str:
            for l in lbl.split(','):
                labels_dict[abst_idx].add(l)

    out = {'index': [], 'sentence': [], 'labels': []}
    for abst_idx in sentence_dict.keys():
        out['index'].append(abst_idx)
        out['sentence'].append(sentence_dict[abst_idx])
        out['labels'].append(','.join(labels_dict[abst_idx]))

    return pd.DataFrame(out)


def remove_sent_wo_labels(df):
    mask = []
    for lbl in df.labels:
        if type(lbl) == str:
            mask.append(True)
        else:
            mask.append(False)
    return df[mask]
