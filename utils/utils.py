import numpy as np
from heapq import nlargest
from sklearn.metrics import precision_recall_fscore_support
import re
import nltk
from nltk.tokenize import RegexpTokenizer
nltk.download('punkt')


re_newline = re.compile(r'\[NEWLINE\]')
re_anon = re.compile(r'\[\*\*.*?\*\*\]')
fix_re = re.compile(r"[^a-z0-9/?.,-:+#]+")
num_re = re.compile(r'[0-9]{2,}')
dash_re = re.compile(r'-+')


def get_label_counts(raw_labels, max_codes=None):
    label_counts = {}
    for label_str in raw_labels:
        label_list = label_str.strip("''").split(',')
        for label in label_list:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

    if max_codes is not None:
        label_counts = nlargest(max_codes, label_counts, key=label_counts.get)

    raw_labels = list(label_counts)
    return label_counts, raw_labels


def get_label_index_mappings(raw_labels):
    label_to_ix = {}
    ix_to_label = {}
    label_ix = 0
    for label_str in raw_labels:
        label_list = label_str.strip("''").split(',')
        for label in label_list:
            if label not in label_to_ix:
                label_to_ix[label] = label_ix
                ix_to_label[label_ix] = label
                label_ix += 1
    return label_to_ix, ix_to_label


def build_vocab(raw_text):
    vocab = set(['<pad>', '<cls>'])
    vocab_lookup = {'<pad>': 0, '<cls>': 1}
    vocab_reverse_lookup = {0: '<pad>', 1: '<cls>'}
    vocab_ix = 2  # Zero reserved for padding

    for note in raw_text:
        words = note.split()
        for w in words:
            if w not in vocab:
                vocab.add(w)
                vocab_lookup[w] = vocab_ix
                vocab_reverse_lookup[vocab_ix] = w
                vocab_ix += 1

    vocab = list(vocab)
    return vocab, vocab_lookup, vocab_reverse_lookup


def fix_word(word, fix_anon=True):
    word = word.lower()
    word = fix_re.sub('-', word)
    if fix_anon:
        word = word.replace('-anon-', '<anon>')
    word = dash_re.sub('-', word)
    return word.strip('-')


def mimic_tokenize(text, fix_anon=True):
    '''Takes in a raw string and returns a list of sentences, each sentence being a list of
       cleaned words.'''
    ret = ''
    for sent in nltk.sent_tokenize(text):
        sent = re_newline.sub('', sent)
        if fix_anon:
            sent = re_anon.sub('-anon-', sent)
        words = nltk.word_tokenize(sent)
        words = [fix_word(word, fix_anon) for word in words]
        words = ' '.join(words)
        if ret == '':
            ret = words
        else:
            ret += ' ' + words
    return ret


def caml_tokenize(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = [t.lower() for t in tokenizer.tokenize(text) if not t.isnumeric()]
    #text = '"' + ' '.join(tokens) + '"'
    return ' '.join(tokens)


def replace_unknown(text, vocab):
    filtered_text = []
    for token in text.split():
        if token in vocab:
            filtered_text.append(token)
        else:
            filtered_text.append('<unk>')
    return ' '.join(filtered_text)

def get_doc_lengths(documents):
    doc_lengths = []
    for doc in documents:
        doc_lengths.append(len(doc.split()))
    return doc_lengths

def remove_tokens(documents, tokens):
    filtered_documents = []
    for doc in documents:
        curr_filtered_document = []
        for token in doc.split():
            if token in tokens:
                continue
            curr_filtered_document.append(token)
        filtered_documents.append(' '.join(curr_filtered_document))
    return filtered_documents
            
def string_to_vec(documents, document_length, word_to_ix):

    doc_matrix = np.zeros((len(documents), document_length))
    doc_lengths = []
    
    for doc_ix, doc in enumerate(documents):
        tokens = doc.split()[:document_length]
        #tokens = doc.split()
        #if len(tokens) > document_length:
        #    tokens = [len(tokens)-document_length:]
        doc_lengths.append(len(tokens))
        for token_ix, token in enumerate(tokens):
            if token in word_to_ix:
                doc_matrix[doc_ix, token_ix] = word_to_ix[token]
            else:
                doc_matrix[doc_ix, token_ix] = word_to_ix['<unk>']
    return doc_matrix, np.array(doc_lengths)

def icd_string_to_vec(labels, label_space_size, label_lookup, sep=','):
    label_matrix = np.zeros((len(labels), label_space_size))
    
    for ix, label_str in enumerate(labels):
        label_list = label_str.strip("''").split(sep)
        for l in label_list:
            if l in label_lookup:
                label_matrix[ix, label_lookup[l]] = 1
            
    return label_matrix

def get_ancestors(code, hierarchy):
    ancestors = set()
    curr_code = code
    while curr_code != '@':
        if curr_code == '719.70':
            curr_code = '719.7'
        ancestors.add(curr_code)
        curr_code = hierarchy[curr_code]
    return list(ancestors)

def expand_labels(labels, hierarchy, sep=','):
    expanded_labels = []
    for label_str in labels:
        if label_str == '':
            expanded_labels.append('')
            continue

        label_list = label_str.strip("''").split(sep)
        expanded_set = set()
        for l in label_list:
            expanded_list = get_ancestors(l, hierarchy)
            for e in expanded_list:
                expanded_set.add(e)
        expanded_labels.append(','.join(list(expanded_set)))
    return expanded_labels

def f1_score(probs, labels, thres, average='micro'):
    '''Returns (precision, recall, F1 score) from a batch of predictions (thresholded probabilities)
       given a batch of labels (for macro-averaging across batches)'''
    preds = (probs >= thres).astype(np.int32)
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average=average,
                                                                 warn_for=())
    return p, r, f

def labels_desc_to_ids(label_desc_file, dicts):
    code_descs = {}
    with open(label_desc_file, 'r') as f:
        for line in f.readlines():
            code = line.split('\t')[0]
            desc = line.split('\t')[1].strip('\n')
            desc = caml_tokenize(desc)
            code_descs[code] = desc
            if code == '719.7':
                code_descs['719.70'] = desc
    
    # Expand vocab
    ix = len(dicts['ind2w'])
    for desc in code_descs.values():
        for token in desc.split():
            if token not in dicts['w2ind']:
                dicts['w2ind'][token] = ix
                dicts['ind2w'][ix] = token
                ix += 1

    descs = []
    num_labels = len(dicts['ind2c'])
    for i in range(num_labels):
        code = dicts['ind2c'][i]
        if code in code_descs:
            desc = code_descs[code]
            descs.append(desc)
        else:
            descs.append('')
            print(f"Note: Description for ICD code {code} not found.")

    max_length = max(get_doc_lengths(descs))
    label_ids, _ = string_to_vec(descs, max_length, dicts['w2ind'])
    label_ids = label_ids.astype(np.int64)

    return label_ids, dicts
