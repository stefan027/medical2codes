import pandas as pd
import json


all_attributes = {
    "model_type": str,
    "gpu": bool,
    "embed_file": str,
    "embed_size": int,
    "update_word_embeddings": bool,
    "embedding_dropout": float,
    "attention_dropout": float,
    "residual_dropout": float,
    "rnn_dropout": float,
    "transformer_dropout": float,
    "learning_rate": float,
    "lm_learning_rate": float,
    "patience": int,
    "attention": bool,
    "embed_labels": bool,
    "filter_size": (int, list),
    "num_filter_maps": int,
    "conv_layer": int,
    "hidden_size": int,
    "attention_type": str,
    "model_weights_save_path": str,
    "model_weights_load_path": str,
    "start_epoch": int,
    "epochs": int,
    "batch_size": int,
    "best_val_score": float,
    "max_doc_length": int,
    "bidirectional": bool,
    "num_layers": int,
    "rnn_cell_type": str,
    "update_transfomer_weights": bool,
    "output_attentions": bool
}


def read_args_from_csv(attributes_csv):
    df = pd.read_csv(attributes_csv)
    
    config_dict = {}
    
    for model_name in df.columns[1:]:
        tmp = {k: v for k, v in zip(df.name.to_list(), df[model_name].to_list())}

        # Ensure None-type is read correctly
        for attr_name, attr_value in tmp.items():
            if type(attr_value) == str:
                if attr_value == 'None':
                    tmp[attr_name] = None
        
        # Ensure data types are correct
        for attr_name, attr_value in tmp.items():
    
            if attr_name in all_attributes:
                attr_dtype = all_attributes[attr_name]
                
                if attr_dtype == bool:
                    tmp[attr_name] = True if tmp[attr_name].lower() == 'true' else False
                elif attr_dtype == int:
                    tmp[attr_name] = int(tmp[attr_name])
                elif attr_dtype == float:
                    tmp[attr_name] = float(tmp[attr_name])
                elif attr_dtype == (int, list):
                    if tmp[attr_name].find(',') == -1:
                        tmp[attr_name] = int(tmp[attr_name])
        config_dict[model_name] = tmp            
        
    attributes_json = '%s.json' % attributes_csv[:len(attributes_csv)-4]                    
    with open(attributes_json, 'w') as f:
        json.dump(config_dict, f, indent=4)
    
    return config_dict
