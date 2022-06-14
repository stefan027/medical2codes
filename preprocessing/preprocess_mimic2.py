########################################################
# Adapted from the code for the following paper:
#
# Adler Perotte, Rimma Pivovarov, Karthik Natarajan, Nicole Weiskopf, Frank Wood, Noemie Elhadad
# Diagnosis Code Assignment: Models and Evaluation Metrics, JAMIA, to apper
#
# Columbia University
# Biomedical Informatics
# Author: Adler Perotte
#
########################################################

import os
import argparse
import numpy as np
import re
from heapq import nlargest

########################################################
# Parse command line arguments
parser = argparse.ArgumentParser(description='Construct datasets from the raw data.')
parser.add_argument('-raw_text', default='MIMIC_RAW_DSUMS', help='File location of the Physionet project raw text.')
parser.add_argument(
    '-raw_icd9', default='ICD-9_v9.xml', help='File location containing the Physionet project raw icd9 code file.'
)
parser.add_argument('-stopwords', default='stopwords.txt', help='File location containing the stopword file.')
parser.add_argument('-out_dir', default='./', help='Where to write output')
args = parser.parse_args()

########################################################
# Preprocess Documents

print("Preprocessing Documents")

# Define stopwords
with open(args.stopwords) as f:
    stopwords = []
    for line in f:
        stopwords.append(line.strip())
stopwords = set(stopwords)

# Term tokenizer
term_pattern = re.compile('[A-Za-z]+')

# Calculate tf and df for tokens
tf = {}
df = {}


def increment_tf_df(tokens_):
    for token in tokens_:
        tf[token] = tf.get(token, 0) + 1.
    uniques = np.unique(tokens_)
    for token in uniques:
        df[token] = df.get(token, 0) + 1.


d = 0
with open(args.raw_text) as f:
    for i, line in enumerate(f):
        raw_dsum = line.split('|')[6]
        
        raw_dsum = re.sub(r'\[[^\]]+\]', ' ', raw_dsum)
        raw_dsum = re.sub(r'admission date:', ' ', raw_dsum, flags=re.I)
        raw_dsum = re.sub(r'discharge date:', ' ', raw_dsum, flags=re.I)
        raw_dsum = re.sub(r'date of birth:', ' ', raw_dsum, flags=re.I)
        raw_dsum = re.sub(r'sex:', ' ', raw_dsum, flags=re.I)
        raw_dsum = re.sub(r'service:', ' ', raw_dsum, flags=re.I)
        raw_dsum = re.sub(r'dictated by:.*$', ' ', raw_dsum, flags=re.I)
        raw_dsum = re.sub(r'completed by:.*$', ' ', raw_dsum, flags=re.I)
        raw_dsum = re.sub(r'signed electronically by:.*$', ' ', raw_dsum, flags=re.I)

        tokens = [token.lower() for token in re.findall(term_pattern, raw_dsum)]
        tokens = [token for token in tokens if token not in stopwords and len(token) > 1]

        increment_tf_df(tokens)

        d += 1


# Evaluate tf idf and top 10k vocab
tfidf = {}
for k in tf:
    tfidf[k] = tf[k] * np.log(float(d)/df[k])

counts = {k: v for k, v in zip(tfidf.keys(), tfidf.values())}
vocab = nlargest(10000, counts, key=counts.get)
vocab = list(vocab)
inv_vocab = {}
for i in range(len(vocab)):
    inv_vocab[vocab[i]] = i

# Write vocab to file
with open(os.path.join(args.out_dir, 'MIMIC_vocabulary2'), 'w') as f:
    for key in vocab:
        f.write(key + '\n')


def convert_to_dict(tokens_):
    token_dict_ = {}
    for token in tokens_:
        token_dict_[token] = token_dict_.get(token, 0) + 1
    return token_dict_


# Tokenize and write raw document to file
with open(args.raw_text) as f:
    with open(os.path.join(args.out_dir, 'MIMIC_FILTERED_DSUMS'), 'w') as f2:
        for i, line in enumerate(f):
            raw_dsum = line.split('|')[6]

            raw_dsum = re.sub(r'\[[^\]]+\]', ' ', raw_dsum)
            raw_dsum = re.sub(r'admission date:', ' ', raw_dsum, flags=re.I)
            raw_dsum = re.sub(r'discharge date:', ' ', raw_dsum, flags=re.I)
            raw_dsum = re.sub(r'date of birth:', ' ', raw_dsum, flags=re.I)
            raw_dsum = re.sub(r'sex:', ' ', raw_dsum, flags=re.I)
            raw_dsum = re.sub(r'service:', ' ', raw_dsum, flags=re.I)
            raw_dsum = re.sub(r'dictated by:.*$', ' ', raw_dsum, flags=re.I)
            raw_dsum = re.sub(r'completed by:.*$', ' ', raw_dsum, flags=re.I)
            raw_dsum = re.sub(r'signed electronically by:.*$', ' ', raw_dsum, flags=re.I)

            tokens = [token.lower() for token in re.findall(term_pattern, raw_dsum)]
            tokens = [token for token in tokens if token not in stopwords and len(token) > 1 and token in inv_vocab]

            # Determine if this DSUM should stay, if so, write to filtered DSUM file
            if len(tokens) > 0:
                # f2.write(line)
                outline = '|'.join(line.split('|')[:6]) + '|' + ' '.join(tokens) + '\n'
                f2.write(outline)

                
# Tokenize and write document counts to file
with open(os.path.join(args.out_dir, 'MIMIC_FILTERED_DSUMS'), 'r') as f:
    with open(os.path.join(args.out_dir, 'MIMIC_term_counts2'), 'w') as f2:
        for i, line in enumerate(f):
            raw_dsum = line.split('|')[6]

            raw_dsum = re.sub(r'\[[^\]]+\]', ' ', raw_dsum)
            raw_dsum = re.sub(r'admission date:', ' ', raw_dsum, flags=re.I)
            raw_dsum = re.sub(r'discharge date:', ' ', raw_dsum, flags=re.I)
            raw_dsum = re.sub(r'date of birth:', ' ', raw_dsum, flags=re.I)
            raw_dsum = re.sub(r'sex:', ' ', raw_dsum, flags=re.I)
            raw_dsum = re.sub(r'service:', ' ', raw_dsum, flags=re.I)
            raw_dsum = re.sub(r'dictated by:.*$', ' ', raw_dsum, flags=re.I)
            raw_dsum = re.sub(r'completed by:.*$', ' ', raw_dsum, flags=re.I)
            raw_dsum = re.sub(r'signed electronically by:.*$', ' ', raw_dsum, flags=re.I)

            tokens = [token.lower() for token in re.findall(term_pattern, raw_dsum)]
            tokens = [token for token in tokens if token not in stopwords and len(token) > 1]

            token_dict = convert_to_dict(tokens)
            
            f2.write(
                str(i) + ',' + ','.join(
                    [str(inv_vocab[k])+':'+str(int(v)) for k, v in token_dict.items() if k in inv_vocab]
                ) + '\n'
            )

##############################################
# Preprocess ICD9 codes

print("Preprocessing ICD9 codes")

# Extract all concepts
with open(args.raw_icd9) as f:
    # concept pool pattern
    concepts_pattern = re.compile(r'<concepts.*?</concepts>', flags=re.DOTALL)
    match_object = concepts_pattern.search(f.read())
    concept_text = match_object.group()
    id_pattern = re.compile(r'id="(.*?)"', flags=re.DOTALL)
    ids = id_pattern.findall(concept_text)
    description_pattern = re.compile(r'lgCommon:entityDescription>(.*?)</lgCommon:entityDescription')
    descriptions = description_pattern.findall(concept_text)
    relationship_pattern = re.compile(r'<lgRel:association id="CHD".*?</lgRel:association>', flags=re.DOTALL)
    match_object2 = relationship_pattern.search(match_object.string)
    relationship_text = match_object2.group()
    parent_pattern = re.compile(r'sourceId="(.*?)"', flags=re.DOTALL)
    parents = parent_pattern.findall(relationship_text)
    child_pattern = re.compile(r'targetId="(.*?)"', flags=re.DOTALL)
    children = child_pattern.findall(relationship_text)


# Write all info on full ICD9 code tree to file
with open(os.path.join(args.out_dir, 'ICD9_descriptions'), 'w') as f:
    f.write('@' + '\t' + 'ICD9 Hierarchy Root' + '\n')
    for code_id, description in zip(ids, descriptions):
        f.write(code_id + '\t' + description + '\n')

# Construct Parent to Child dictionary
PtoC = {}
with open(os.path.join(args.out_dir, 'ICD9_parent_child_relations'), 'w') as f:
    for parent, child in zip(parents, children):
        if parent != 'V-ICD9CM':
            f.write(parent + '\t' + child + '\n')
            if parent in PtoC:
                PtoC[parent].append(child)
            else:
                PtoC[parent] = [child]

# Remove all procedures from the tree
root = '@'
procedure_root = '00-99.99'

# Delete as a child of the root
PtoC['@'].remove(procedure_root)

# Go through the PtoC dict and remove entries recursively
to_remove = [procedure_root]
while len(to_remove) > 0:
    new_to_remove = []
    for code in to_remove:
        if code in PtoC:
            new_to_remove.extend(PtoC[code])
            del PtoC[code]
    to_remove = new_to_remove

# The entire code set minus the procedures
code_set = set()
for code in PtoC:
    code_set.add(code)
    code_set.update(PtoC[code])

# Write out the ICD9 parent child relationships without procedures
with open(os.path.join(args.out_dir, 'ICD9_parent_child_relations'), 'r') as f:
    with open(os.path.join(args.out_dir, 'ICD9_parent_child_relations_no_proc'), 'w') as f2:
        for line in f:
            parent, child = line.strip().split('\t')
            if parent in code_set and child in code_set:
                f2.write(line)


# Write out the ICD9 descriptions without procedures
code_descriptions = {}
with open(os.path.join(args.out_dir, 'ICD9_descriptions'), 'r') as f:
    with open(os.path.join(args.out_dir, 'ICD9_descriptions_no_proc'), 'w') as f2:
        for line in f:
            code, description = line.strip().split('\t')
            if code in code_set:
                f2.write(line)
                code_descriptions[code] = description

# Construct an array or arrays containing ancestors for each code
root = '@'
ancestors = {}
children = PtoC[root]
for child in children:
    ancestors[child] = ancestors.get(child, []) + [root]
while len(children) > 0:
    new_children = []
    for child in children:
        for gc in PtoC.get(child, []):
            ancestors[gc] = ancestors.get(gc, []) + [child] + ancestors[child]
            new_children.append(gc)
    children = new_children
for i in ancestors:
    ancestors[i] = ancestors[i] + [i]

# Collect all codes that are actually used in the dataset
used_codes = {}
codes_plus_ancestors = set()
total_count = 0
with open(os.path.join(args.out_dir, 'MIMIC_FILTERED_DSUMS'), 'r') as f:
    for line in f:
        total_count += 1
        codes = line.strip().split('|')[5].strip('"').split(',')
        for code in codes:
            if code == '719.70':
                code = '719.7'
            used_codes[code] = used_codes.get(code, 0) + 1
            codes_plus_ancestors.update(ancestors[code])

# Generate indices for codes and parents of those codes
active_code_list = list(codes_plus_ancestors)
code_index_lookup = {}
for i, code in enumerate(active_code_list):
    code_index_lookup[code] = i
    if code == "@":
        root_i = i
        with open('ROOT', 'w') as f:
            f.write(str(i) + '\n')

# Write out the ICD9 code-index mapping
with open(os.path.join(args.out_dir, 'MIMIC_ICD9_mapping'), 'w') as f:
    for i, code in enumerate(active_code_list):
        f.write(code + '\t' + str(i) + '\t' + code_descriptions[code] + '\n')

# Write out the final parent to child relationship file
with open(os.path.join(args.out_dir, 'ICD9_parent_child_relations_no_proc'), 'r') as f:
    with open(os.path.join(args.out_dir, 'MIMIC_parentTochild'), 'w') as f2:
        for line in f:
            line = line.strip().split('\t')
            parent = line[0]
            child = line[1]
            if parent in codes_plus_ancestors and child in codes_plus_ancestors:
                f2.write(str(code_index_lookup[parent]) + '|' + str(code_index_lookup[child]) + '\n')


# Generate a dataset of codes where codes are replaced by indices (no ancestors)
codes_no_ancestors = []
with open(os.path.join(args.out_dir, 'MIMIC_FILTERED_DSUMS'), 'r') as f:
    with open(os.path.join(args.out_dir, 'MIMIC_ICD9_codes_no_ancestors'), 'w') as f2:
        for i, line in enumerate(f):
            codes_no_ancestors.append(0)
            line = line.strip().split('|')
            doc_id = line[1]
            codes = line[5].strip('"').split(',')
            f2.write(str(i))
            for code in codes:
                if code == '719.70':
                    code = '719.7'  # Weird irregularity
                if code in code_index_lookup:
                    f2.write('|' + str(code_index_lookup[code]))
                    codes_no_ancestors[-1] += 1
            f2.write('\n')


# Same as above but with ancestors
codes_with_ancestors = []
with open(os.path.join(args.out_dir, 'MIMIC_FILTERED_DSUMS'), 'r') as f:
    with open(os.path.join(args.out_dir, 'MIMIC_ICD9_codes'), 'w') as f2:
        for i, line in enumerate(f):
            codes_with_ancestors.append(0)
            line = line.strip().split('|')
            doc_id = line[1]
            codes = line[5].strip('"').split(',')
            f2.write(str(i))
            set_to_write = set()
            for leaf in codes:
                if leaf == '719.70':
                    leaf = '719.7'  # An exception
                all_anc = ancestors[leaf]
                set_to_write.update(list(all_anc))
            for code in set_to_write:
                f2.write('|' + str(code_index_lookup[code]))
                codes_with_ancestors[-1] += 1
            f2.write('\n')

##############################################
# Create training and testing sets

print("Creating training and testing sets")

total_n = 22815
training_split = 9.
testing_split = 1.
shuffle = False

training_n = int(total_n*training_split/float(training_split+testing_split))
testing_n = total_n - training_n

if shuffle:
    perm = np.random.permutation(total_n)
else:
    perm = np.arange(total_n)
training_ind = set(list(perm[:training_n]))
testing_ind = set(list(perm[training_n:]))

with open(os.path.join(args.out_dir, 'MIMIC_term_counts2'), 'r') as data_file:
    with open(os.path.join(args.out_dir, 'training_text.data'), 'w') as training_file:
        with open(os.path.join(args.out_dir, 'testing_text.data'), 'w') as testing_file:
            for i, line in enumerate(data_file):
                if i in training_ind:
                    training_file.write(line)
                else:
                    testing_file.write(line)

with open(os.path.join(args.out_dir, 'MIMIC_ICD9_codes'), 'r') as data_file:
    with open(os.path.join(args.out_dir, 'training_codes.data'), 'w') as training_file:
        with open(os.path.join(args.out_dir, 'testing_codes.data'), 'w') as testing_file:
            for i, line in enumerate(data_file):
                if i in training_ind:
                    training_file.write(line)
                else:
                    testing_file.write(line)


with open(os.path.join(args.out_dir, 'MIMIC_ICD9_codes_no_ancestors'), 'r') as data_file:
    with open(os.path.join(args.out_dir, 'training_codes_na.data'), 'w') as training_file:
        with open(os.path.join(args.out_dir, 'testing_codes_na.data'), 'w') as testing_file:
            for i, line in enumerate(data_file):
                if i in training_ind:
                    training_file.write(line)
                else:
                    testing_file.write(line)

with open(os.path.join(args.out_dir, 'training_indices.data'), 'w') as f:
    f.write('\n'.join([str(x) for x in np.sort(list(training_ind))]) + '\n')

with open(os.path.join(args.out_dir, 'testing_indices.data'), 'w') as f:
    f.write('\n'.join([str(x) for x in np.sort(list(testing_ind))]) + '\n')


# Create final files for training, validation and testing (NOT IN PEROTTE'S ORIGINAL CODE)

validation_size = 0.1

with open(os.path.join(args.out_dir, 'training_indices.data'), 'r') as f:
    train_indices = [int(idx.strip('\n')) for idx in f.readlines()]

with open(os.path.join(args.out_dir, 'testing_indices.data'), 'r') as f:
    test_indices = [int(idx.strip('\n')) for idx in f.readlines()]

total_size = len(train_indices) + len(test_indices)
validation_size = int(total_size*validation_size)

validation_indices = train_indices[-validation_size:]
train_indices = train_indices[:-validation_size]

train_indices = set(train_indices)
validation_indices = set(validation_indices)
test_indices = set(test_indices)

with open(os.path.join(args.out_dir, 'mimic2_train.txt'), 'w') as f:
    f.write("patient_id|admission_id|date|note_type|unknown_field|Label|TEXT\n")
with open(os.path.join(args.out_dir, 'mimic2_valid.txt'), 'w') as f:
    f.write("patient_id|admission_id|date|note_type|unknown_field|Label|TEXT\n")
with open(os.path.join(args.out_dir, 'mimic2_test.txt'), 'w') as f:
    f.write("patient_id|admission_id|date|note_type|unknown_field|Label|TEXT\n")
    
with open(os.path.join(args.out_dir, 'MIMIC_FILTERED_DSUMS'), 'r') as input_file:
    for idx, line in enumerate(input_file):
        if idx in train_indices:
            with open(os.path.join(args.out_dir, 'mimic2_train.txt'), 'a') as f:
                f.write(line.replace('"', '').replace('[NEWLINE]', ''))
        elif idx in validation_indices:
            with open(os.path.join(args.out_dir, 'mimic2_valid.txt'), 'a') as f:
                f.write(line.replace('"', '').replace('[NEWLINE]', ''))
        else:
            with open(os.path.join(args.out_dir, 'mimic2_test.txt'), 'a') as f:
                f.write(line.replace('"', '').replace('[NEWLINE]', ''))
