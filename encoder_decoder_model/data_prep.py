"""python data_prep.py --thermo_train_data thermo_train --thermo_val_data thermo_val --thermo_test_data thermo_test --meso_train_data meso_train --meso_val_data meso_val --meso_test_data meso_test """

import argparse
from log_configs import *
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence


#importing the data input file
parser = argparse.ArgumentParser()
parser.add_argument("--thermo_train_data", help="Specify if working with Mesophilic/Thermophilic", type=str)
parser.add_argument("--thermo_val_data", help="Specify if working with Mesophilic/Thermophilic", type=str)
parser.add_argument("--thermo_test_data", help="Specify if working with Mesophilic/Thermophilic", type=str)
parser.add_argument("--meso_train_data", help="Specify if working with Mesophilic/Thermophilic", type=str)
parser.add_argument("--meso_val_data", help="Specify if working with Mesophilic/Thermophilic", type=str)
parser.add_argument("--meso_test_data", help="Specify if working with Mesophilic/Thermophilic", type=str)



args = parser.parse_args()
train_meso_file = pd.read_csv(args.meso_train_data,header=None)
train_thermo_file = pd.read_csv(args.thermo_train_data,header=None)
valid_meso_file = pd.read_csv(args.meso_val_data,header=None)
valid_thermo_file = pd.read_csv(args.thermo_val_data,header=None)
test_meso_file = pd.read_csv(args.meso_test_data,header=None)
test_thermo_file = pd.read_csv(args.thermo_test_data,header=None)

print(train_meso_file)

"""## 1. Data preprocessing""" 
# Rostlab/prot_bert requires that the AA are separated between each other with a space, so will transform our input file in this format

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


"""## 2. Dataset"""

### 1. Load Data
"""The dataset is strcutured in 3 files, already splitted into training, validation, test set (created by using .py)"""

logging.info('----- DATASET STRUCTURE ----- ')

# Load train data
train_meso_file = pd.read_csv(train_meso_file,header=None)
# Display 5 samples from the test data
print(train_meso_file.sample(5))
logging.info('TRAIN SET STRUCTURE MESO EXAMPLE, # PAIRS:{}'.format(len(train_meso_file)))


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
