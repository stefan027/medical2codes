import argparse


def read_cli_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-mimic_version", type=int, help="2 or 3")
    parser.add_argument("-data_dir", type=str, help="path to data directory")
    parser.add_argument("-config", type=str, help="path to config file")
    parser.add_argument("-e", "--train_expanded_labels", action="store_true", default=False)
    parser.add_argument("-vocab_file", type=str, help="path to vocab file", default=None)
    parser.add_argument("-eval_only", action="store_true", default=False)
    parser.add_argument("-eval_data", type=str, help="train, val or test", default="test")
    parser.add_argument("-model_path", type=str, help="path to saved model weights", default=None)
    parser.add_argument("-shuffle", action="store_true", help="Shuffle data before each epoch", default=False)
    parser.add_argument("-max_codes", type=int, help="Limit to n most frequent codes", default=None)
    parser.add_argument("-random_seed", type=int, help="Fix random seed", default=None)
    parser.add_argument("-xlnet_base_model", type=str, help="Path to pretrained XLNet", default=None)
    parser.add_argument("-output_dir", type=str, help="Location to save outputs", default='./')

    return parser.parse_args()
