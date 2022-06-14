def initialise_model(args, num_labels, dicts, label_ids=None):
    if args.model_type == 'CNN':
        from models.cnn import MultiCNN
        return MultiCNN(args, num_labels, dicts, label_ids)

    if args.model_type == 'ResCNN':
        from models.cnn import MultiResCNN
        return MultiResCNN(args, num_labels, dicts, label_ids)

    if args.model_type == 'RNN':
        from models.rnn import RNNTextClassifier
        return RNNTextClassifier(args, num_labels, dicts, label_ids)

    if args.model_type == 'XLNet':
        from models.multilabel_xlnet import MultilabelXLNet
        return MultilabelXLNet(args=args, num_labels=num_labels, dicts=dicts, label_ids=label_ids)
