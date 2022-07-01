import argparse
import shutil
import os
from log_configs import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from collections import Counter
import random
import time

import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer,BertModel, BertAdapterModel, AutoConfig,  BertAdapterModel, BertModel,AutoModel
from transformers import AdamW, get_linear_schedule_with_warmup

from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report, confusion_matrix, precision_recall_curve, matthews_corrcoef


from transformers.modeling_utils import PreTrainedModel , PretrainedConfig, PretrainedConfig, PreTrainedModel



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


def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def numel(m: torch.nn.Module, only_trainable: bool = False):
    """
    returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)


def evaluate_roc_valdata(probs, y_true,val_threshold=None):
    """
    - Print AUC and accuracy on the test set
    - Plot ROC
    @params    probs (np.array): an array of predicted probabilities with shape (len(y_true), 2)
    @params    y_true (np.array): an array of the true values with shape (len(y_true),)
    """
    logging.info('*** EVALUATION ON VALIDATION DATA ***')
    print('--Tuning the inference threshold using ROC if not specified one')
    preds = probs[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true, preds)
    lr_precision, lr_recall, _ = precision_recall_curve(y_true, preds)
    threshold = thresholds[np.argmin(np.abs(fpr+tpr-1))]
    
    roc_auc = auc(fpr, tpr)
    logging.info(f'AUC: {roc_auc:.4f}')

    if val_threshold:
        threshold = val_threshold
        logging.info(f'Desidered Threshold : {threshold:.4f}')
    else:
        threshold = threshold 
        logging.info(f'Tuned Threshold: {threshold:.4f}')
       
    # Get accuracy over the val set
    y_pred = np.where(preds >= threshold, 1, 0)	
    accuracy = accuracy_score(y_true, y_pred)
    MC = matthews_corrcoef(y_true, y_pred)
    logging.info(f'Matthews Correlation Coefficient computed after applying the tuned/selected threshold : {MC}')
    logging.info(f'Accuracy: {accuracy*100:.2f}%')

    print('--Creating Threshold plot and ROC plot (until last bestmodel if training from checkpoints)')
    # Plot thresholds value (https://www.yourdatateacher.com/2021/06/14/are-you-still-using-0-5-as-a-threshold/)
    plt.title('Threshold Tuning')
    plt.scatter(thresholds,np.abs(fpr+tpr-1))
    plt.xlabel("Threshold")
    plt.ylabel("|FPR + TPR - 1|")
    plt.show()
    plt.savefig(str('threshold_tuning_valdata_'+ str(threshold) +'.png'))
    plt.clf()

    # Plot ROC AUC
    plt.title('Receiver Operating Characteristic Val Data')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(str('ROC_valdata_' + str(threshold)+'.png'))
    plt.clf()


    # plot the precision-recall curves
    plt.title('Precision-Recall curves on Val Data')
    no_skill = len(y_true[y_true==1]) / len(y_true)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(lr_recall, lr_precision, marker='.', label='Classifier')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    # show the plot
    plt.savefig(str('precision_recall_val' + str(threshold) +'.png'))
    plt.clf()

    # Creating classification report
    logging.info('--Classification report for VAL DATA--')
    logging.info(classification_report(y_true,y_pred))
    
    unique_label = np.unique([y_true, y_pred])
    cm = pd.DataFrame(
    confusion_matrix(y_true, y_pred, labels=unique_label), 
    index=['true:{:}'.format(x) for x in unique_label], 
    columns=['pred:{:}'.format(x) for x in unique_label]
    )

    logging.info(cm)

    return True


def evaluate_roc_testdata(probs, y_true,val_threshold):
    """
    - Print AUC and accuracy on the test set
    - Plot ROC
    @params    probs (np.array): an array of predicted probabilities with shape (len(y_true), 2)
    @params    y_true (np.array): an array of the true values with shape (len(y_true),)
    """
    logging.info('*** EVALUATION ON TEST DATA USING VALIDATION THRESHOLD ***')
    logging.info(f'Applying the tuned threshold on val data: {val_threshold:.4f}')


    preds=probs[:, 1]
    preds = np.where(preds >= val_threshold, 1, 0)
    fpr, tpr, thresholds = roc_curve(y_true, preds)
    lr_precision, lr_recall, _ = precision_recall_curve(y_true, preds)
    test_threshold = thresholds[np.argmin(np.abs(fpr+tpr-1))]
    logging.info(f'Threshold estimated on test set:  {test_threshold:.4f}')
    
    roc_auc = auc(fpr, tpr)
    logging.info(f'AUC: {roc_auc:.4f}')

    # computing MC Coefficients
    MC = matthews_corrcoef(y_true, preds)
    logging.info(f'Matthews Correlation Coefficient computed after applying the tuned/selected threshold : {MC}')


    # Get accuracy over the test set
    accuracy = accuracy_score(y_true, preds)
    logging.info(f'Accuracy on test set: {accuracy*100:.2f}%')


    print('--Creating ROC plot (on best model from checkpoints)')
    # Plot thresholds value (https://www.yourdatateacher.com/2021/06/14/are-you-still-using-0-5-as-a-threshold/)
    plt.title('Threshold on test set')
    plt.scatter(thresholds,np.abs(fpr+tpr-1))
    plt.xlabel("Threshold")
    plt.ylabel("|FPR + TPR - 1|")
    plt.show()
    plt.savefig(str('threshold_testdata_' + str(val_threshold) +'.png'))
    plt.clf()


    # Plot ROC AUC
    plt.title('Receiver Operating Characteristic Test Data')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(str('ROC_testdata_' + str(val_threshold) + '.png'))
    plt.clf()


    # plot the precision-recall curves
    plt.title('Precision-Recall curves on Test Data')
    no_skill = len(y_true[y_true==1]) / len(y_true)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(lr_recall, lr_precision, marker='.', label='Classifier')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    # show the plot
    plt.savefig(str('precision_recall_test_' + str(val_threshold) +'.png'))
    plt.clf()

    # Creating classification report
    logging.info('Classification report for TEST DATA using validation threshold or defined by us:')
    logging.info(classification_report(y_true,preds))
    unique_label = np.unique([y_true, preds])
    cm = pd.DataFrame(
    confusion_matrix(y_true, preds, labels=unique_label), 
    index=['true:{:}'.format(x) for x in unique_label], 
    columns=['pred:{:}'.format(x) for x in unique_label]
    )

    
    #cm = confusion_matrix(y_true,y_pred)
    logging.info(cm)


    return True


