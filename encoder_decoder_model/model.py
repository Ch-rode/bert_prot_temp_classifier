"""python model.py --thermo_train_data thermo_train --thermo_val_data thermo_val --thermo_test_data thermo_test --meso_train_data meso_train --meso_val_data meso_val --meso_test_data meso_test  --start_epochs 0   --n_epochs 1  --learning_rate 0.0000003 --pred_threshold 0.90 --batch_size 1 --num_workers 0 """


#en=meso
""" 0. Saving model and Checkpoint setup"""
import shutil
import os

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    function created to save checkpoint, the latest one and the best one. 
    This creates flexibility: either you are interested in the state of the latest checkpoint or the best checkpoint.
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)


def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()


# create the checkpoint and best model directories
ckp_dir = 'checkpoint'
parent_dir = os.getcwd()
best_dir = 'best_model'
print(parent_dir)

ckp_dir= os.path.join(parent_dir, ckp_dir)

try:
    os.mkdir(ckp_dir) 
except FileExistsError:
        pass

best_dir= os.path.join(parent_dir, best_dir)

try:
    os.mkdir(best_dir) 
except FileExistsError:
        pass


""" 0.1 Argparse setup and arguments"""
import argparse
from log_configs import *

parser = argparse.ArgumentParser()
parser.add_argument("--thermo_train_data", help="Specify if working with Mesophilic/Thermophilic", type=str)
parser.add_argument("--thermo_val_data", help="Specify if working with Mesophilic/Thermophilic", type=str)
parser.add_argument("--thermo_test_data", help="Specify if working with Mesophilic/Thermophilic", type=str)
parser.add_argument("--meso_train_data", help="Specify if working with Mesophilic/Thermophilic", type=str)
parser.add_argument("--meso_val_data", help="Specify if working with Mesophilic/Thermophilic", type=str)
parser.add_argument("--meso_test_data", help="Specify if working with Mesophilic/Thermophilic", type=str)
parser.add_argument("--start_epochs", help="Specify if working with Mesophilic/Thermophilic", type=int)
parser.add_argument("--n_epochs", help="Specify if working with Mesophilic/Thermophilic", type=int)
parser.add_argument("--learning_rate", help="Specify if working with Mesophilic/Thermophilic", type=float)
#parser.add_argument("--meso_max_len", help="Specify if working with Mesophilic/Thermophilic", type=int)
#parser.add_argument("--thermo_max_len", help="Specify if working with Mesophilic/Thermophilic", type=int)
parser.add_argument("--pred_threshold", help="Specify if working with Mesophilic/Thermophilic", type=float)
parser.add_argument("--batch_size", help="Specify if working with Mesophilic/Thermophilic", type=int)
parser.add_argument("--num_workers", help="Specify if working with Mesophilic/Thermophilic", type=int)


args = parser.parse_args()

""" python model.py --thermo_train_data thermo_train --thermo_val_data thermo_val --thermo_test_data thermo_test --meso_train_data meso_train --meso_val_data meso_val ---meso_test_data meso_test  --start_epochs 0   --n_epochs 1  --learning_rate 0.0000003 --pred_threshold 0.90 --batch_size 1 --num_workers 0 """


lr = args.learning_rate #5e-06
#max_len = args.max_len #512
threshold = args.pred_threshold #0.9
batch_size = args.batch_size
num_workers = args.num_workers
start_epochs = args.start_epochs
n_epochs = args.n_epochs # 10
enc_maxlength = 20
dec_maxlength = 20

vocabsize = 30
max_length = 20


import pandas as pd
train_meso_file = pd.read_csv(args.meso_train_data,header=None)
train_thermo_file = pd.read_csv(args.thermo_train_data,header=None)
valid_meso_file = pd.read_csv(args.meso_val_data,header=None)
valid_thermo_file = pd.read_csv(args.thermo_val_data,header=None)
test_meso_file = pd.read_csv(args.meso_test_data,header=None)
test_thermo_file = pd.read_csv(args.thermo_test_data,header=None)

print(train_meso_file)

# Rostlab/prot_bert requires that the AA are separated between each other with a space

train_meso_file[0]= [" ".join("".join(sample.split())) for sample in train_meso_file[0]]
train_thermo_file[0]=[" ".join("".join(sample.split())) for sample in train_thermo_file[0]]
valid_meso_file[0]=[" ".join("".join(sample.split())) for sample in valid_meso_file[0]]
valid_thermo_file[0]=[" ".join("".join(sample.split())) for sample in valid_thermo_file[0]]
test_meso_file[0]=[" ".join("".join(sample.split())) for sample in test_meso_file[0]]
test_thermo_file[0]=[" ".join("".join(sample.split())) for sample in test_thermo_file[0]]


train_meso_file.to_csv(args.meso_train_data,index=None,header=False)
train_thermo_file.to_csv(args.thermo_train_data,index=None,header=False)
valid_meso_file.to_csv(args.meso_val_data,index=None,header=False)
valid_thermo_file.to_csv(args.thermo_val_data,index=None,header=False)
test_meso_file.to_csv(args.meso_test_data,index=None,header=False)
test_thermo_file.to_csv(args.thermo_test_data,index=None,header=False)


#check
print(valid_thermo_file.head())

train_meso_file = args.meso_train_data
train_thermo_file = args.thermo_train_data
valid_meso_file = args.meso_val_data
valid_thermo_file = args.thermo_val_data
test_meso_file = args.meso_test_data
test_thermo_file = args.thermo_test_data 


logging.info('----- MODEL RUN SELECTED PARAMETERS: ----- ')

logging.info('* LEARNING RATE {}'.format(lr))

logging.info('* N EPOCHS:{}'.format(n_epochs))

logging.info('* BATCH SIZE {}'.format(batch_size))

logging.info('* MAX LEN {}'.format(max_length))

logging.info('* PREDICTION THRESHOLD:{}'.format(threshold))

"""# A - Setup """

import sys
import json


## 1. Load Essential Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from collections import Counter
import random
import time


from transformers import BertTokenizerFast
from transformers import BertModel, BertForMaskedLM, BertConfig, EncoderDecoderModel, BertLMHeadModel, EncoderDecoderConfig
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence

from transformers import BertTokenizer,BertModel
from transformers import AdamW, get_linear_schedule_with_warmup

from sklearn.metrics import accuracy_score, roc_curve, auc


## 2. Set Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

"""## 2. Dataset"""

### 2.1. Load Data
"""The dataset is strcutured in 3 files, already splitted into training, validation, test set (created by using .py)"""

logging.info('----- DATASET STRUCTURE ----- ')

# Load train data
#train_meso_file = pd.read_csv(train_meso_file,header=None)
# Display 5 samples from the test data
#print(train_meso_file.sample(5))
logging.info('TRAIN SET STRUCTURE MESO EXAMPLE, # PAIRS:{}'.format(len(train_meso_file)))


# The custom dataloader class takes as input the respective EN and DE files, the corresponding tokenizers and max lengths. 
# Once we have tokenized the texts, we first define a __getitem__() to return the corresponding pair of EN and DE texts. 
# We then define the collate function to perform padding and generate masks for each batch.

class TranslationDataset(data.Dataset):

    def __init__(self, inp_file, targ_file, inp_tokenizer, targ_tokenizer, inp_maxlength, targ_maxlength):

        self.inp_tokenizer = inp_tokenizer
        self.targ_tokenizer = targ_tokenizer
        self.inp_maxlength = inp_maxlength
        self.targ_maxlength = targ_maxlength

        print("Loading and Tokenizing the data ...")
        self.encoded_inp = []
        self.encoded_targ = []

        # Read the EN lines
        num_inp_lines = 0
        with open(inp_file, "r") as ef:
            for line in ef:
                enc = self.inp_tokenizer.encode(line.strip(), add_special_tokens=True, max_length=self.inp_maxlength)
                self.encoded_inp.append(torch.tensor(enc))
                num_inp_lines += 1

        # read the DE lines
        num_targ_lines = 0
        with open(targ_file, "r") as df:
            for line in df:
                enc = self.targ_tokenizer.encode(line.strip(), add_special_tokens=True, max_length=self.targ_maxlength)
                self.encoded_targ.append(torch.tensor(enc))
                num_targ_lines += 1

        assert (num_inp_lines==num_targ_lines), "Mismatch in THERMO and MESO lines"
        print("Read", num_inp_lines, "lines from THERMO and MESO files.")

    def __getitem__(self, offset):
        meso = self.encoded_inp[offset]
        thermo = self.encoded_targ[offset]

        return meso, meso.shape[0], thermo, thermo.shape[0]

    def __len__(self):
        return len(self.encoded_inp)

    def collate_function(self, batch):

        (inputs, inp_lengths, targets, targ_lengths) = zip(*batch)

        padded_inputs = self._collate_helper(inputs, self.inp_tokenizer)
        padded_targets = self._collate_helper(targets, self.targ_tokenizer)

        max_inp_seq_len = padded_inputs.shape[1]
        max_out_seq_len = padded_targets.shape[1]

        input_masks = [[1]*l + [0]*(max_inp_seq_len-l) for l in inp_lengths]
        target_masks = [[1]*l + [0]*(max_out_seq_len-l) for l in targ_lengths]

        input_tensor = padded_inputs.to(torch.int64)
        target_tensor = padded_targets.to(torch.int64)
        input_masks = torch.Tensor(input_masks)
        target_masks = torch.Tensor(target_masks)

        return input_tensor, input_masks, target_tensor, target_masks

    def _collate_helper(self, examples, tokenizer):
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

"""## 3. Set up GPU for training"""

logging.info('----- GPU INFORMATION: ----- ')

if torch.cuda.is_available():       
    device = torch.device("cuda")
    logging.info(f'There are {str(torch.cuda.device_count())} GPU(s) available.')
    for i in range(torch.cuda.device_count()):
        logging.info(f'GPU Device name: {torch.cuda.get_device_name(int(i))}, ')

else:
    logging.info('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

print("Using device:", device)

"""## 4. Tokenization"""

logging.info('----- TOKENIZATION: ----- ')

# Load the BERT tokenizer

tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert', do_lower_case=False)

tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token
logging.info("CLS token: {}, SEP token: {}".format(tokenizer.cls_token,tokenizer.sep_token))


# Load the tokenizers

meso_tokenizer = tokenizer 
thermo_tokenizer = tokenizer 


# Init the dataset


train_dataset = TranslationDataset(train_meso_file, train_thermo_file, meso_tokenizer, thermo_tokenizer, enc_maxlength, dec_maxlength)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, \
                                                drop_last=True, num_workers=1, collate_fn=train_dataset.collate_function)

valid_dataset = TranslationDataset(valid_meso_file, valid_thermo_file, meso_tokenizer, thermo_tokenizer, enc_maxlength, dec_maxlength)
valid_dataloader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, \
                                                drop_last=True, num_workers=1, collate_fn=valid_dataset.collate_function)

"""### 6. Model

### 6.1. Create Bert2Bert """


print("Loading models ..")



encoder_config = BertConfig(vocab_size = vocabsize,
                    max_position_embeddings = max_length+64, # this shuold be some large value
                    num_attention_heads = 16,
                    num_hidden_layers = 30,
                    hidden_size = 1024,
                    type_vocab_size = 1,
                    )

encoder = BertModel(config=encoder_config)

vocabsize = 30
max_length = 20
decoder_config = BertConfig(vocab_size = vocabsize,
                    max_position_embeddings = max_length+64, # this shuold be some large value
                    num_attention_heads = 16,
                    num_hidden_layers = 30,
                    hidden_size = 1024,
                    type_vocab_size = 1,
                    is_decoder=True,
                    add_cross_attention=True)    # Very Important

#decoder = BertForMaskedLM(config=decoder_config)
decoder = BertLMHeadModel(config=decoder_config)


# Define encoder decoder model
model = EncoderDecoderModel(encoder=encoder, decoder=decoder,)

model.to(device)
#logging.info("Model Structure : {}".format(model))


def count_parameters(mdl):
    return sum(p.numel() for p in mdl.parameters() if p.requires_grad)

logging.info(f'The encoder has {count_parameters(encoder):,} trainable parameters')
logging.info(f'The decoder has {count_parameters(decoder):,} trainable parameters')
logging.info(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.NLLLoss(ignore_index=thermo_tokenizer.pad_token_id)

num_train_batches = len(train_dataloader)
num_valid_batches = len(valid_dataloader)


def compute_loss(predictions, targets):
    """Compute our custom loss"""
    predictions = predictions[:, :-1, :].contiguous()
    targets = targets[:, 1:]

    rearranged_output = predictions.view(predictions.shape[0]*predictions.shape[1], -1)
    rearranged_target = targets.contiguous().view(-1)

    loss = criterion(rearranged_output, rearranged_target)

    return loss


    
def eval_model(val_dataloader):
    model.eval()
    epoch_loss = 0

    for i, (meso_input, meso_masks, thermo_output, thermo_masks) in enumerate(val_dataloader):

        optimizer.zero_grad()

        meso_input = meso_input.to(device)
        thermo_output = thermo_output.to(device)
        meso_masks = meso_masks.to(device)
        thermo_masks = thermo_masks.to(device)

        labels = thermo_output.clone()

        out = model(input_ids=meso_input, attention_mask=meso_masks,
                                        decoder_input_ids=thermo_output, decoder_attention_mask=thermo_masks,labels=labels)

        prediction_scores = out[1]
        predictions = F.log_softmax(prediction_scores, dim=2)
        loss = compute_loss(predictions, thermo_output)
        epoch_loss += loss.item()

    val_loss= (epoch_loss / num_valid_batches) #Mean validation loss

    return val_loss



# MAIN TRAINING LOOP

def train(model, train_dataloader, val_dataloader, valid_loss_min_input, checkpoint_path, best_model_path, start_epochs, epochs, evaluation=True):

    # Start training loop
    logging.info("--Start training...\n")

    # Initialize tracker for minimum validation loss
    valid_loss_min = valid_loss_min_input 

    for epoch_i in range(start_epochs,epochs):
        # Start training loop
        logging.info("--Start training...\n")

        # Put the model into the training mode
        model.train()

        # Print the header of the result table
        logging.info((f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10}  | {'Elapsed':^9}"))

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        epoch_loss, batch_loss, batch_counts = 0, 0, 0

    
        for i, (meso_input, meso_masks, thermo_output, thermo_masks) in enumerate(train_dataloader):

            step=i

            batch_counts +=1
            optimizer.zero_grad()

            meso_input = meso_input.to(device)
            thermo_output = thermo_output.to(device)
            meso_masks = meso_masks.to(device)
            thermo_masks = thermo_masks.to(device)

            labels = thermo_output.clone()
            out = model(input_ids=meso_input, attention_mask=meso_masks,
                                            decoder_input_ids=thermo_output, decoder_attention_mask=thermo_masks,labels=labels)
            prediction_scores = out[1]
            predictions = F.log_softmax(prediction_scores, dim=2)
            loss = compute_loss(predictions, thermo_output)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_loss += loss.item()
            epoch_loss += loss.item()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 500 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                logging.info(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        avg_train_loss = epoch_loss / num_train_batches

        logging.info("Mean epoch loss: {}".format(avg_train_loss))

        logging.info("-"*70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss = eval_model(val_dataloader)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            
            logging.info(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} |  | {time_elapsed:^9.2f}")

            logging.info("-"*70)
        logging.info("\n")


        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch_i + 1,
            'valid_loss_min': val_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        
        # save checkpoint
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)
    
        ## TODO: save the model if validation loss has decreased
        if val_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,val_loss))
            # save checkpoint as best model
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            valid_loss_min = val_loss


def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


set_seed(42)    # Set seed for reproducibility
train(model = model, train_dataloader = train_dataloader, val_dataloader = valid_dataloader, start_epochs = start_epochs, epochs = n_epochs, valid_loss_min_input = np.Inf, evaluation = True, checkpoint_path = r"./checkpoint/current_checkpoint.pt", best_model_path = r"./best_model/best_model.pt")





#######################################################################################################################################
################################################ INFERENCE ############################################################################
#######################################################################################################################################
checkpoint = torch.load(r"./best_model/best_model.pt")
try:
    checkpoint.eval()
except AttributeError as error:
    print (error)
### 'dict' object has no attribute 'eval'

model.load_state_dict(checkpoint['state_dict'])
### now you can evaluate it
model.eval()
#model, optimizer, start_epoch, valid_loss_min = load_ckp(r"./best_model/best_model.pt", model, optimizer)




logging.info('TESTING THE MODEL OUT')

input_text="M D I P K D R F Y T K T H E W A L P E G D T V L V G I T D Y A Q D A L G D V V Y V E L P E V G R T V E A G E A V A V V E S V K T A S D I Y A P V A G E V V E V N L A L E K S P E L I N Q D P Y G E G W I F R L R P R D M A D L D G L L D A S G Y Q E A L E A G A"
#generated = model.generate(input_ids, decoder_start_token_id=model.config.decoder.pad_token_id)
print(len(input_text))
logging.info('Mesophilic protein to be translated into Thermophilic: {}'.format(input_text))


def generate_translation(batch):
    # cut off at BERT max length 512
    inputs = tokenizer(batch, padding="max_length", truncation=True, max_length=model.config.max_length, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    outputs = model.generate(input_ids, attention_mask=attention_mask,max_length=model.config.max_length,decoder_start_token_id=model.config.decoder.pad_token_id)
    print(outputs)
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True,padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    
    print(output_str)
    return output_str 


logging.info("Translation output {}".format(generate_translation(input_text)))



"""
def predict(row, model):
    # convert row to data
    row = torch.tensor(row)
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat

#out = model(input_ids=meso_input, attention_mask=meso_masks)
 #           prediction_scores = out[1]
   #         predictions = F.log_softmax(prediction_scores, dim=2)
#yhat = predict(input_text, model)
#print(yhat)


#input_ids = torch.tensor(tokenizer.encode(input_text)).unsqueeze(0)
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
print(input_ids)
output_ids = model.generate(input_ids.to('cuda'),decoder_start_token_id=model.config.decoder.pad_token_id)
print(output_ids)
output_text = tokenizer.decode(output_ids[0])
logging.info('TEST {}'.format(output_text))
#print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))

" CUDA_VISIBLE_DEVICES=1,2,4, python thermo_bert_classifier.py --train_data TRAIN_1m_1tcl --val_data VAL_1m_1tcl --test_data TEST_1m_1tcl --start_epochs 0 --n_epochs 5 --learning_rate 5e-06 --max_len 768 --pred_threshold 0.9 --batch_size 16 --num_workers 8"


def predict(row, model):
    # convert row to data
    row = torch.tensor(row)
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat

#yhat = predict(input_text, model)
#print(yhat)
"""