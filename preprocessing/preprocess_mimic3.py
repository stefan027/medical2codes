import os
import numpy as np
import pandas as pd
import csv
import operator
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
from scipy.sparse import csr_matrix
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("-data_dir", type=str, help="Directory where MIMIC-III data is stored")
parser.add_argument("-icd_file", type=str, help="Path to ICD-9 source file")
parser.add_argument("-vocab_min", type=int, help="Min token frequency to be included in vocab")

cli_args = parser.parse_args()
data_dir = cli_args.data_dir
icd_file = cli_args.icd_file
vocab_min = cli_args.vocab_min

# Combine diagnosis and procedure codes and reformat them
# The codes in MIMIC-III are given in separate files for procedures and diagnoses,
# and the codes are given without periods, which might lead to collisions if we naively
# combine them. So we have to add the periods back in the right place.

dfproc = {"ROW_ID": [], "SUBJECT_ID": [], "HADM_ID": [], "SEQ_NUM": [], "ICD9_CODE": []}
with open(os.path.join(data_dir, 'PROCEDURES_ICD.csv'), 'r') as f:
    for row in f.readlines()[1:]:
        values = row.strip().split(',')
        # Skip over records for no ICD9 codes
        if values[3] == '':
            continue
        dfproc["ROW_ID"].append(int(values[0]))
        dfproc["SUBJECT_ID"].append(int(values[1]))
        dfproc["HADM_ID"].append(int(values[2]))
        dfproc["SEQ_NUM"].append(int(values[3]))
        dfproc["ICD9_CODE"].append(values[4].strip('"'))
dfproc = pd.DataFrame(dfproc)

dfdiag = {"ROW_ID": [], "SUBJECT_ID": [], "HADM_ID": [], "SEQ_NUM": [], "ICD9_CODE": []}
with open(os.path.join(data_dir, 'DIAGNOSES_ICD.csv'), 'r') as f:
    for row in f.readlines()[1:]:
        values = row.strip().split(',')
        # Skip over records for no ICD9 codes
        if values[3] == '':
            continue
        dfdiag["ROW_ID"].append(int(values[0]))
        dfdiag["SUBJECT_ID"].append(int(values[1]))
        dfdiag["HADM_ID"].append(int(values[2]))
        dfdiag["SEQ_NUM"].append(int(values[3]))
        dfdiag["ICD9_CODE"].append(values[4].strip('"'))
dfdiag = pd.DataFrame(dfdiag)


def reformat(code_, is_diag):
    """
    Put a period in the right place because the MIMIC-3 data files exclude them.
    Generally, procedure code_s have dots after the first two digits,
    while diagnosis code_s have dots after the first three digits.
    """
    code_ = ''.join(code_.split('.'))
    if is_diag:
        if code_.startswith('E'):
            if len(code_) > 4:
                code_ = code_[:4] + '.' + code_[4:]
        else:
            if len(code_) > 3:
                code_ = code_[:3] + '.' + code_[3:]
    else:
        if len(code_) > 2:
            code_ = code_[:2] + '.' + code_[2:]
    return code_


dfdiag['absolute_code'] = dfdiag.apply(lambda row_: str(reformat(str(row_[4]), True)), axis=1)
dfproc['absolute_code'] = dfproc.apply(lambda row_: str(reformat(str(row_[4]), False)), axis=1)
dfcodes = pd.concat([dfdiag, dfproc])

icd_new = pd.read_csv(icd_file)

icd_set = set()
icd_desc = {}
icd_parent = {}

for classID, Description, Parent in zip(icd_new['Class ID'], icd_new['Preferred Label'], icd_new['Parents']):
    code = classID.split('/')[-1]
    if Description in ['Entity', 'Event']:
        continue
    
    parent = Parent.split('/')[-1]

    icd_set.add(code)
    icd_desc[code] = Description
    icd_parent[code] = parent

mask = np.zeros((len(dfcodes)), dtype=bool)
for i, code in enumerate(dfcodes.absolute_code.to_list()):
    if code not in icd_set:
        if code != 'nan' and code != '719.70':
            mask[i] = True
unknown = set(dfcodes[mask].absolute_code.to_list())

unknown_parent_map = {}
unknown_desc_map = {}

for code in unknown:
    dot_ix = code.find('.')
    pre_dot = code[:dot_ix]
    pst_dot = code[dot_ix+1:]
    if len(pst_dot) == 2:
        parent = f'{pre_dot}.{pst_dot[0]}'
    else:
        parent = pre_dot
    unknown_parent_map[code] = parent
    unknown_desc_map[code] = icd_desc[parent]

with open(os.path.join(data_dir, 'ICD9_descriptions_updated'), 'w') as f:
    f.write('@\tICD9 Hierarchy Root\n')
    for k, v in icd_desc.items():
        f.write(f'{k}\t{v}\n')
    for k, v in unknown_desc_map.items():
        f.write('{k}\t{v}\n')
        
with open(os.path.join(data_dir, 'ICD9_parent_child_relations_updated'), 'w') as f:
    for k, v in icd_parent.items():
        f.write('{k}\t{v}\n')
    for k, v in unknown_parent_map.items():
        f.write('{k}\t{v}\n')

# Sort by SUBJECT_ID and HADM_ID
dfcodes = dfcodes.sort_values(['SUBJECT_ID', 'HADM_ID'])
dfcodes.to_csv(
    os.path.join(data_dir, 'ALL_CODES.csv'),
    index=False,
    columns=['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'absolute_code'],
    header=['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'ICD9_CODE']
)


# Tokenize and preprocess raw text
# This will:
#     - Select only discharge summaries and their addenda
#     - Remove punctuation and numeric-only tokens, removing 500 but keeping 250mg
#     - Lowercase all tokens

# retain only alphanumeric
tokenizer = RegexpTokenizer(r'\w+')


def write_discharge_summaries(out_file):
    notes_file = f'{data_dir}/NOTEEVENTS.csv'
    print("processing notes file")
    with open(notes_file, 'r') as csvfile:
        with open(out_file, 'w') as outfile:
            print(f"writing to {out_file}")
            outfile.write(','.join(['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'TEXT']) + '\n')
            notereader = csv.reader(csvfile)
            # skip over header
            next(notereader)
            for line in tqdm(notereader):
                category = line[6]
                if category == "Discharge summary":
                    note = line[10]
                    # tokenize, lowercase and remove numerics
                    tokens = [t.lower() for t in tokenizer.tokenize(note) if not t.isnumeric()]
                    text = '"' + ' '.join(tokens) + '"'
                    outfile.write(','.join([line[1], line[2], line[4], text]) + '\n')
    return out_file


disch_full_file = write_discharge_summaries(out_file=os.path.join(data_dir, "disch_full.csv"))

# Sort by SUBJECT_ID and HADM_ID
df = pd.read_csv(os.path.join(data_dir, "disch_full.csv"))
df = df.sort_values(['SUBJECT_ID', 'HADM_ID'])
df.to_csv(os.path.join(data_dir, "disch_full.csv"), index=False)


# Consolidate labels with set of discharge summaries

# Filter out HADM_ID without discharge summaries
hadm_ids = set(df['HADM_ID'])
with open(os.path.join(data_dir, 'ALL_CODES.csv'), 'r') as lf:
    with open(os.path.join(data_dir, 'ALL_CODES_filtered.csv'), 'w') as of:
        w = csv.writer(of, dialect='unix')
        w.writerow(['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE', 'ADMITTIME', 'DISCHTIME'])
        r = csv.reader(lf)
        # skip over header
        next(r)
        for i, row in enumerate(r):
            hadm_id = int(row[2])
            if hadm_id in hadm_ids:
                w.writerow(row[1:3] + [row[-1], '', ''])

# Filter out HADM_ID without labels
df1 = pd.read_csv(os.path.join(data_dir, "disch_full.csv"))
df2 = pd.read_csv(os.path.join(data_dir, "ALL_CODES_filtered.csv"))
hadm_ids = set(df1.HADM_ID).intersection(set(df2.HADM_ID))
mask = []
for hadm_id in df1.HADM_ID.to_list():
    if hadm_id in hadm_ids:
        mask.append(True)
    else:
        mask.append(False)
df1[mask].to_csv(os.path.join(data_dir, "disch_full.csv"), index=False)

# Concatenate the labels with the notes data and split using the saved splits
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def concat_data(labelsfile, notes_file):
    """
    INPUTS:
        labelsfile: sorted by hadm id, contains one label per line
        notes_file: sorted by hadm id, contains one note per line
    """
    with open(labelsfile, 'r') as lf_:
        print("CONCATENATING")
        with open(notes_file, 'r') as notesfile:
            outfilename = os.path.join(data_dir, 'notes_labeled.csv')
            with open(outfilename, 'w') as outfile:
                writer = csv.writer(outfile, dialect='unix')
                writer.writerow(['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS'])

                labels_gen = next_labels(lf_)
                notes_gen = next_notes(notesfile)

                for j, (subj_id, text, hadm_id_) in enumerate(notes_gen):
                    if j % 10000 == 0:
                        print(str(j) + " done")
                    cur_subj, cur_labels, cur_hadm = next(labels_gen)

                    if cur_hadm == hadm_id_:
                        writer.writerow([subj_id, str(hadm_id_), text, ';'.join(cur_labels)])
                    else:
                        print("couldn't find matching hadm_id. data is probably not sorted correctly")
                        break
                    
    return outfilename


def split_data(labeledfile, base_name):
    print("SPLITTING")
    # create and write headers for train, dev, test
    train_name = f'{base_name}_train_split.csv'
    dev_name = f'{base_name}_dev_split.csv'
    test_name = f'{base_name}_test_split.csv'
    train_file = open(train_name, 'w')
    dev_file = open(dev_name, 'w')
    test_file = open(test_name, 'w')
    train_file.write(','.join(['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS']) + "\n")
    dev_file.write(','.join(['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS']) + "\n")
    test_file.write(','.join(['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS']) + "\n")

    hadm_ids_ = {}

    # read in train, dev, test splits
    for splt_ in ['train', 'dev', 'test']:
        hadm_ids_[splt_] = set()
        with open(f'{data_dir}/{splt_}_full_hadm_ids.csv', 'r') as f_:
            for line in f_:
                hadm_ids_[splt_].add(line.rstrip())

    with open(labeledfile, 'r', newline='\n') as lf_:
        reader = csv.reader(lf_)
        next(reader)
        j = 0
        for row_ in reader:
            # filter text, write to file according to train/dev/test split
            if j % 10000 == 0:
                print(str(j) + " read")
            hadm_id_ = row_[1]

            if hadm_id_ in hadm_ids_['train']:
                train_file.write(','.join(row_) + "\n")
            elif hadm_id_ in hadm_ids_['dev']:
                dev_file.write(','.join(row_) + "\n")
            elif hadm_id_ in hadm_ids_['test']:
                test_file.write(','.join(row_) + "\n")

            j += 1

        train_file.close()
        dev_file.close()
        test_file.close()
    return train_name, dev_name, test_name


def next_labels(labelsfile):
    """
        Generator for label sets from the label file
    """
    labels_reader = csv.reader(labelsfile)
    # skip over header
    next(labels_reader)

    first_label_line = next(labels_reader)

    cur_subj = int(first_label_line[0])
    cur_hadm = int(first_label_line[1])
    cur_labels = [first_label_line[2]]

    for row_ in labels_reader:
        subj_id_ = int(row_[0])
        hadm_id_ = int(row_[1])
        code_ = row_[2]
        # keep reading until you hit a new hadm id
        if hadm_id_ != cur_hadm or subj_id_ != cur_subj:
            yield cur_subj, cur_labels, cur_hadm
            cur_labels = [code_]
            cur_subj = subj_id_
            cur_hadm = hadm_id_
        else:
            # add to the labels and move on
            cur_labels.append(code_)
    yield cur_subj, cur_labels, cur_hadm


def next_notes(notesfile):
    """
    Generator for notes from the notes file
    This will also concatenate discharge summaries and their addenda, which have the same subject and hadm id
    """
    nr = csv.reader(notesfile)
    # skip over header
    next(nr)

    first_note = next(nr)

    cur_subj = int(first_note[0])
    cur_hadm = int(first_note[1])
    cur_text = first_note[3]
    
    for row_ in nr:
        subj_id_ = int(row_[0])
        hadm_id_ = int(row_[1])
        text = row_[3]
        # keep reading until you hit a new hadm id
        if hadm_id_ != cur_hadm or subj_id_ != cur_subj:
            yield cur_subj, cur_text, cur_hadm
            cur_text = text
            cur_subj = subj_id_
            cur_hadm = hadm_id_
        else:
            # concatenate to the discharge summary and move on
            cur_text += " " + text
    yield cur_subj, cur_text, cur_hadm


labeled = concat_data(
    os.path.join(data_dir, 'ALL_CODES_filtered.csv'),
    os.path.join(data_dir, 'disch_full.csv')
)


# Create train/dev/test splits

fname = os.path.join(data_dir, 'notes_labeled.csv')
base_name_ = os.path.join(data_dir, "disch")  # for output
tr, dv, te = split_data(fname, base_name=base_name_)


# Build vocabulary from training data
# This function reads a sorted training dataset and builds a vocabulary of terms of given size
# Output: txt file with vocab words
# Drops any token not appearing in at least vocab_min notes


def build_vocab(vocab_minimum, infile, vocab_filename):
    """
    INPUTS:
        vocab_minimum: how many documents a word must appear in to be kept
        infile: (training) data file to build vocabulary from
        vocab_filename: name for the file to output
    """
    with open(infile, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # skip over header
        next(reader)

        # 0. read in data
        print("reading in data...")
        # holds number of terms in each document
        note_numwords = []
        # indices where notes start
        note_inds = [0]
        # indices of discovered words
        indices = []
        # holds a bunch of ones
        data = []
        # keep track of discovered words
        vocab = {}
        # build lookup table for terms
        num2term = {}
        # preallocate array to hold number of notes each term appears in
        note_occur = np.zeros(400000, dtype=int)
        for row_ in reader:
            text = row_[2]
            numwords = 0
            for term in text.split():
                # put term in vocab if it's not there. else, get the index
                index = vocab.setdefault(term, len(vocab))
                indices.append(index)
                num2term[index] = term
                data.append(1)
                numwords += 1
            # record where the next note starts
            note_inds.append(len(indices))
            indset = set(indices[note_inds[-2]:note_inds[-1]])
            # go thru all the word indices you just added, and add to the note occurrence count for each of them
            for ind in indset:
                note_occur[ind] += 1
            note_numwords.append(numwords)
        # clip trailing zeros
        note_occur = note_occur[note_occur > 0]

        # turn vocab into a list so indexing doesn't get fd up when we drop rows
        vocab_list = np.array([word for word, ind in sorted(vocab.items(), key=operator.itemgetter(1))])

        # 1. create sparse document matrix
        c = csr_matrix((data, indices, note_inds), dtype=int).transpose()

        # 2. remove rows with less than 3 total occurrences
        print("removing rare terms")
        # inds holds indices of rows corresponding to terms that occur in < 3 documents
        inds = np.nonzero(note_occur >= vocab_minimum)[0]
        print(str(len(inds)) + " terms qualify out of " + str(c.shape[0]) + " total")

        # drop those rows
        vocab_list = vocab_list[inds]

        print("writing output")
        with open(vocab_filename, 'w') as vocab_file:
            for word in vocab_list:
                vocab_file.write(word + "\n")


build_vocab(vocab_min, tr, os.path.join(data_dir, 'vocab.csv'))

# Sort each data split by length for batching
for splt in ['train', 'dev', 'test']:
    filename = f'{data_dir}/disch_{splt}_split.csv'
    df = pd.read_csv(filename)
    df['length'] = df.apply(lambda row_: len(str(row_['TEXT']).split()), axis=1)
    df = df.sort_values(['length'])
    df.to_csv(f'{data_dir}/{splt}_full.csv', index=False)


# Write final output files
def process_data(in_fp, out_fp):
    with open(in_fp, 'r') as f1:
        lines = f1.readlines()
        with open(out_fp, 'w') as f2:
            f2.write('patient_id|admission_id|Label|TEXT\n')
            for line in lines[1:]:
                patient_id = int(line.split(',')[0])
                admission_id = int(line.split(',')[1])
                text = line.split(',')[2]
                label = line.split(',')[3]
                label_list = label.strip().split(';')
                label = ','.join(label_list)
                f2.write(f"{patient_id}|{admission_id}|'{label}'|{text}\n")
    

process_data(os.path.join(data_dir, 'train_full.csv'), os.path.join(data_dir, 'mimic3_train.csv'))
process_data(os.path.join(data_dir, 'dev_full.csv'), os.path.join(data_dir, 'mimic3_valid.csv'))
process_data(os.path.join(data_dir, 'test_full.csv'), os.path.join(data_dir, 'mimic3_test.csv'))
