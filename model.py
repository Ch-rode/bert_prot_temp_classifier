from distutils.command.config import config
from utilities import *

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

#Create the BertClassfier class

class BertClassifier(PreTrainedModel):
    """Bert Model for Classification Tasks."""
    config_class = AutoConfig
    def __init__(self,config, freeze_bert=True): #tuning only the head
        """
         @param    bert: a BertModel object
         @param    classifier: a torch.nn.Module classifier
         @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        #super(BertClassifier, self).__init__()
        super().__init__(config)

        # Instantiate BERT model
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        #configuration = AutoConfig.from_pretrained("Rostlab/prot_bert_bfd")
        #configuration = BertConfig.from_pretrained('Rostlab/prot_bert_bfd')
        self.bert = BertModel.from_pretrained('Rostlab/prot_bert_bfd',config=config)
        #self.bert = BertModel(configuration)
        #configuration = self.bert.config
        self.D_in = 1024 #hidden size of Bert
        self.H = 512
        self.D_out = 2
 
        # Instantiate the classifier head with some one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.D_in, 512),
            nn.Tanh(),
            nn.Linear(512, self.D_out),
            nn.Tanh()
        )
 
         # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = True #FALSE: If you want to freeze part of your model and train the rest, you can set requires_grad of the parameters you want to freeze to False.


    def forward(self, input_ids, attention_mask):


         # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                             attention_mask=attention_mask)
         
         # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]
 
         # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)
 
        return logits





#Create the BertClassfier class
class BertClassifierAdapter(PreTrainedModel):
    """Bert Model for Classification Tasks."""
    config_class = AutoConfig
    def __init__(self,config, freeze_bert=True): #tuning only the head
        """
         @param    bert: a BertModel object
         @param    classifier: a torch.nn.Module classifier
         @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        #super(BertClassifier, self).__init__()
        super().__init__(config)

        # Instantiate BERT model
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        self.bert = BertAdapterModel.from_pretrained('Rostlab/prot_bert_bfd',config=config)
        self.D_in = 1024 #hidden size of Bert
        self.H = 512
        self.D_out = 2
        

        # Add a new adapter
        self.bert.add_adapter("sequence_adapter",set_active=True)
        self.bert.train_adapter(["sequence_adapter"])

 
        # Instantiate the classifier head with some one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.D_in, 512),
            nn.Tanh(),
            nn.Linear(512, self.D_out),
            nn.Tanh()
        )
 
 #If you want to freeze part of your model and train the rest, you can set requires_grad of the parameters you want to freeze to False.

         # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = True


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


     
def initialize_model(device,train_dataloader,epochs,lr,adapter=None):
    """ Initialize the Bert Classifier, the optimizer and the learning rate scheduler."""
    configuration=AutoConfig.from_pretrained('Rostlab/prot_bert_bfd')
    if adapter == True:
        # Instantiate Bert Classifier
        logging.info(' --- Training with Adapters ---')
        bert_classifier = BertClassifierAdapter(config=configuration,freeze_bert=False)
    else:
        bert_classifier = BertClassifier(config=configuration,freeze_bert=False)
        

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


