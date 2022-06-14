class Config:
    
    def __init__(self, config_dict):
        
        self.model_type = config_dict['model_type'] if 'model_type' in config_dict else 'CNN'

        self.gpu = config_dict['gpu'] if 'gpu' in config_dict else False

        self.embed_file = config_dict['embed_file'] if 'embed_file' in config_dict else None
        self.embed_size = config_dict['embed_size'] if 'embed_size' in config_dict else 100
        self.update_word_embeddings = config_dict['update_word_embeddings'] \
            if 'update_word_embeddings' in config_dict else True
        
        self.embedding_dropout = config_dict['embedding_dropout'] if 'embedding_dropout' in config_dict else 0.3
        self.attention_dropout = config_dict['attention_dropout'] if 'attention_dropout' in config_dict else 0.3
        
        self.learning_rate = config_dict['learning_rate'] if 'learning_rate' in config_dict else 0.003
        self.patience = config_dict['patience'] if 'patience' in config_dict else 5
        
        self.attention = config_dict['attention'] if 'attention' in config_dict else True
        self.embed_labels = config_dict['embed_labels'] if 'embed_labels' in config_dict else False

        if self.model_type in ('CNN', 'ResCNN'):
            self.filter_size = config_dict['filter_size'] if 'filter_size' in config_dict else 4
            self.num_filter_maps = config_dict['num_filter_maps'] if 'num_filter_maps' in config_dict else 500
            self.conv_layer = config_dict['conv_layer'] if 'conv_layer' in config_dict else 1
            self.residual_dropout = config_dict['residual_dropout'] if 'residual_dropout' in config_dict else 0.3
        elif self.model_type == 'RNN':
            self.hidden_size = config_dict['hidden_size'] if 'hidden_size' in config_dict else False
            self.attention_type = config_dict['attention_type'] if 'attention_type' in config_dict else 'general'
            self.bidirectional = config_dict['bidirectional'] if 'bidirectional' in config_dict else False
            self.num_layers = config_dict['num_layers'] if 'num_layers' in config_dict else 1
            self.rnn_cell_type = config_dict['rnn_cell_type'] if 'rnn_cell_type' in config_dict else 'gru'
            self.rnn_dropout = config_dict['rnn_dropout'] if 'rnn_dropout' in config_dict else 0.0
        elif self.model_type == 'XLNet':
            self.hidden_size = config_dict['hidden_size'] if 'hidden_size' in config_dict else False
            self.update_transfomer_weights = config_dict['update_transfomer_weights'] \
                if 'update_transfomer_weights' in config_dict else True
            self.transformer_dropout = config_dict['transformer_dropout'] if \
                'transformer_dropout' in config_dict else 0.1
            self.lm_learning_rate = config_dict['lm_learning_rate'] if 'lm_learning_rate' in config_dict else 2E-5
            self.output_attentions = config_dict['output_attentions'] if 'output_attentions' in config_dict else False
            self.xlnet_base_model = None

        self.model_weights_save_path = None
        self.model_weights_load_path = None

        self.start_epoch = config_dict['start_epoch'] if 'start_epoch' in config_dict else 0
        self.epochs = config_dict['epochs'] if 'epochs' in config_dict else 50
        self.batch_size = config_dict['batch_size'] if 'batch_size' in config_dict else 16
        self.best_val_score = config_dict['best_val_score'] if 'best_val_score' in config_dict else 0
        self.max_doc_length = config_dict['max_doc_length'] if 'max_doc_length' in config_dict else 2500
