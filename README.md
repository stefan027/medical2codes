# Automatic assignment of diagnosis codes to free-form text medical notes

This is the code repository for research done as part of my M.Sc. thesis which can be accessed at the [Stellenbosch University library](http://hdl.handle.net/10019.1/123654).

Some of the codebase was adapted from previous work on this dataset, in particular:
- [Perotte et al. (2014)](https://archive.physionet.org/works/ICD9CodingofDischargeSummaries) (requires Physionet access - see below under Data for more information)
- [Mullenbach et al. (2018)](https://github.com/jamesmullenbach/caml-mimic)
- [Li et al. (2019)](https://github.com/foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network)

The code was written for research purposes and, admittedly, contains some inelegant and hastily put-together bits of code to run some experiments :smile:; however, please let me know if you find any scientific errors.

## Thesis Abstract

Clinical coding is the process of describing and categorising healthcare episodes according to standardised ontologies. The coded data have important downstream applications, including population morbidity studies, health systems planning and reimbursement. Clinical codes are generally assigned based on information contained in free-form text clinical notes by specialist human coders. This process is expensive, time-consuming, subject to human error and burdens scarce clinical human resources with administrative roles. An accurate automatic coding system can alleviate these problems.

Clinical coding is a challenging task for machine learning systems. The source texts are often long, has a highly specialised vocabulary, contains non-standard clinician shorthand and the code sets can contain tens-of-thousands of codes.

In our research we review previous work on clinical auto-coding systems and perform an empirical analysis of widely used and current state-of-the-art machine learning approaches to the problem. We propose a novel attention mechanism that takes the text description of clinical codes into account. We also construct a small pre-trained transformer model that achieves state-of-the-art performance on the MIMIC II and III ICD-9 auto-coding tasks. To the best of our knowledge, it is the first successful application of a pre-trained transformer model on this task.

## Requirements
The code was tested on Python 3.8. The following dependencies must be installed (code was tested on the versions in parentheses). Refer to the `requirements.txt` file for more details.
- numpy (1.22.2)
- pandas (1.3.5)
- gensim (4.2.0)
- sklearn (0.24.0)
- pytorch (1.11.0)
- transformers (4.19.4)
- nltk (3.7)

## Getting started

Refer to the Colab notebook for code samples: [Colab Getting Started](https://colab.research.google.com/drive/1d0PufGGzkE3p1eTHsuiIzsLqKCo9nGp0?usp=sharing)

## Data

The data is available for research purposes on [Physionet](https://physionet.org/) for credentialed users. The credentialing process involves completing an online certification in human research subject protection and [HIPAA](https://en.wikipedia.org/wiki/Health_Insurance_Portability_and_Accountability_Act) regulations, and signing the data use agreement.

When training on MIMIC II, the code looks for `mimic2_train.txt`, `mimic2_valid.txt` and `mimic2_test.txt`. When training on MIMIC III, the code looks for `mimic3_train.csv`, `mimic3_valid.csv` and `mimic3_test.csv`. The location of the data directory can be specified from the command line (see below).

**See the README in the _preprocessing_ directory for data preparation steps**

## Pretrained language models
Pre-trained word2vec models and XLNet-256 pre-trained on MIMIC III can be downloaded with `wget`:
```bash
wget -O pretrained_word2vec.zip --no-check-certificate "https://onedrive.live.com/download?cid=05F351192BFCCB72&resid=5F351192BFCCB72%2116059&authkey=AC3miUydIYitp58"
wget -O xlnet_mimic.zip --no-check-certificate "https://onedrive.live.com/download?cid=05F351192BFCCB72&resid=5F351192BFCCB72%2116542&authkey=AFfxLSYriNtZEUM"
```

## Configuration files
Hyperparameters and other configuration settings can be specified in a csv or json file. Examples for config files for RNNs, CNNs and XLNet-256 can be found in `./config`. The code allows for multiple models to be specified in a single config file. The models will be run consecutively. The model weights and test set predictions will be saved based on the `model_name` parameter specified in the config file.

## Training
To fine-tune an XLNet-256 model, run:
```bash
python run.py -mimic_version <2-or-3> \
  -data_dir <path-to-preprocessed-data> \
  -xlnet_base_model <path-to-pretrained-language-model> \
  -config <path-to-config-file>
```
To train the RNN and CNN-based models, run:
```bash
python run.py -mimic_version <2-or-3> \
  -data_dir <path-to-preprocessed-data> \
  -xlnet_base_model <path-to-pretrained-language-model>
```
Optional arguments:
- To specify where outputs are saved use `-output_dir <location-to-save-outputs>`
- To train on the expanded label set (i.e. including ancestor nodes) add `-e`
- To specify a limited vocabulary include `-vocab_file <path-to-vocab-file>`
- To load model weights from a checkpoint include `-model_path <path-to-model-weights-file>`. The code accepts a Pytorch state dict only.
- To shuffle the training documents before each epoch add `-shuffle`
- To limit the number of labels to the top N codes add `-max_codes <N>`
- To fix the random seed add `-random_seed <seed>`
- To run evaluation only add `-eval_only`
    - The default is to run evaluation on the test split
    - To run evaluation on the validation add `-eval_data val`
    - To run evaluation on the training add `-eval_data train`
