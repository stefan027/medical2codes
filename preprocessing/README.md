# Preprocessing 

## MIMIC II
We adopted the preprocessing performed by Perotte et al. on the MIMIC II datasets. The data can be downloaded by Physionet credentialed users from this archive: https://archive.physionet.org/works/ICD9CodingofDischargeSummaries/files/

### Requirements

1. The raw data file from Physionet (MIMIC_RAW_DSUMS)
2. The ICD-9 descriptions (version used by Perotte et al.) can be downloaded here: https://onedrive.live.com/download?cid=05F351192BFCCB72&resid=5F351192BFCCB72%2146023&authkey=AGW0Ob8Cuf4Hgm0
3. File with list of stopwords excluded by Perotte et al. can be downloaded here: https://onedrive.live.com/download?cid=05F351192BFCCB72&resid=5F351192BFCCB72%2145981&authkey=ALZ687GrI7t76zU

### Running the preprocessing script

```
python preprocess_mimic2.py \
    -raw_text <path to MIMIC_RAW_DSUMS> \
    -raw_icd9 <path to ICD-9 descriptions file> \
    -stopwords <path to stopwords.txt> \
    -out_dir <directory to write preprocessed data to>
```

## MIMIC III

Physionet credentialed users can download the MIMIC III database here: https://physionet.org/content/mimiciii/1.4/

### Requirements

1. The following MIMIC III files must be in the same directory:
    - PROCEDURES_ICD.csv
    - DIAGNOSES_ICD.csv
    - ALL_CODES.csv
    - NOTEEVENTS.csv

2. ICD-9 tree can be downloaded from here: https://onedrive.live.com/download?cid=05F351192BFCCB72&resid=5F351192BFCCB72%2131680&authkey=ACvLiFSA6-J0XmU

3. IDs for train, validation and test splits must be in the same location as the MIMIC III data. The files can be downloaded here: https://onedrive.live.com/download?cid=05F351192BFCCB72&resid=5F351192BFCCB72%21198607&authkey=AM7fkJhz0nz6DaA

    - train_full_hadm_ids.csv
    - test_full_hadm_ids.csv
    - dev_full_hadm_ids.csv

```
python preprocess_mimic3.py \
  -data_dir <path to directory with input data> \
  -icd_file <path to ICD-9 descriptions file> \
  -vocab_min <exclude tokens with fewer than vocab_min occurences>
```
