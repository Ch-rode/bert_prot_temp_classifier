from utils import *

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert', do_lower_case=False)


# Create a function to tokenize a set of texts
def preprocessing_for_bert(data,MAX_LEN):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence:
    for sent in data:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            sent,                           # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,             # Max length to truncate/pad
            pad_to_max_length=True,         # Pad sentence to max length
            #return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True,     # Return attention mask
            truncation=True     
            )
        
        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks



class BertTempProtConfig(PretrainedConfig):
    #AutoConfig.from_pretrained("Rostlab/prot_bert_bfd")
    model_type="BertTempProtClassifier"
    def __init__(
        self,
        vocab_size=30,
        hidden_size=1024,
        num_hidden_layers=30,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=40000,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        transformer_version="4.17.0",
        torch_dtype="float32",
        **kwargs
    ):

        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout


bert_config=BertTempProtConfig()


#Create the BertClassfier class
class BertTempProtClassifier(PreTrainedModel):
    """Bert Model for Classification Tasks."""
    config_class = BertTempProtConfig
    def __init__(self,config=bert_config, freeze_bert=None,adapter=None,mode=None): #tuning only the head
        """
         @param    bert: a BertModel object
         @param    classifier: a torch.nn.Module classifier
         @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        #super(BertClassifier, self).__init__()
        super().__init__(config)


        self.D_in = 1024 #hidden size of Bert
        self.H = 512
        self.D_out = 2

        
        if adapter == 'True':
            if mode == 'train':
                self.bert = BertAdapterModel.from_pretrained('Rostlab/prot_bert_bfd',config=config)
                # Add a new adapter
                self.bert.add_adapter("tem_prot_adapter",set_active=True)
                self.bert.train_adapter(["tem_prot_adapter"])
            else:
                self.bert = BertAdapterModel(config=config)
                self.bert.load_adapter("./best_model_hugginface/final_adapter")
                self.bert.set_active_adapters('tem_prot_adapter')

        else:
            if mode == 'train':
                self.bert = BertModel.from_pretrained('Rostlab/prot_bert_bfd',config=config)
            else:
                self.bert = BertModel(config=config)

 
        # Instantiate the classifier head with some two-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.D_in, 512),
            nn.Tanh(),
            nn.Linear(512, self.D_out),
            nn.Tanh()
        )
 
        # Freeze the BERT model
        if freeze_bert == 'True':
            for param in self.bert.parameters():
                param.requires_grad = False
                #logging.info('freeze_bert: {}'.format(freeze_bert)) 
                #logging.info('param.requires_grad: {}'.format(param.requires_grad))
        if freeze_bert == 'False':
            for param in self.bert.parameters():
                param.requires_grad = True
                #logging.info('freeze_bert: {}'.format(freeze_bert)) 
                #logging.info('param.requires_grad: {}'.format(param.requires_grad))
                

    def forward(self, input_ids, attention_mask):
        ''' Feed input to BERT and the classifier to compute logits.
         @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                       max_length)
         @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                       information with shape (batch_size, max_length)
         @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                       num_labels) '''
         # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                             attention_mask=attention_mask)
         
         # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]
 
         # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)
 
        return logits



def initialize_model(device,train_dataloader,epochs,lr,adapter=None,fine_tuning=None,mode=None):
    """ Initialize the Bert Classifier, the optimizer and the learning rate scheduler."""
    
    if adapter == 'True':
        # Instantiate Bert Classifier
        logging.info(' --- Training with Adapters ---')
        logging.info('Not fine-tuning Bert, freezing Bert parameters')
        bert_classifier = BertTempProtClassifier(adapter='True',mode=mode)
        #logging.info(bert_classifier)
    else:
        if fine_tuning == 'True':
            logging.info('Fine-tuning Bert, unfreezing Bert parameters')
            bert_classifier = BertTempProtClassifier(freeze_bert='False',mode=mode)
            #logging.info(bert_classifier)
        else:
            logging.info('Not fine-tuning Bert, freezing Bert parameters')
            bert_classifier = BertTempProtClassifier(freeze_bert='True',mode=mode)
            #logging.info(bert_classifier)
    

    logging.info('Number of trainable parameters: {}'.format(sum(p.numel() for p in bert_classifier.parameters() if p.requires_grad)))
    print('Number of trainable parameters: {}'.format(sum(p.numel() for p in bert_classifier.parameters() if p.requires_grad)))
    logging.info('CHECK PARAMETERS TRAINABLE {}'.format(numel(bert_classifier, only_trainable=True)))
    logging.info('Number of total parameters {}'.format(numel(bert_classifier, only_trainable=False)))

    # Tell PyTorch to run the model on GPU
    bert_classifier = bert_classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=lr,    # Default learning rate
                      eps=1e-8    # Default epsilon value
                      )

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler

