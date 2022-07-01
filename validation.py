
from utils import *
from testing import predict
from training import preprocessing_for_bert
from model import BertTempProtClassifier



def validation(val_data,val_batch_size, device, num_workers, adapter=None,val_threshold=None,):
    logging.info( ' *** VALIDATION MODE ***')
    
    if adapter == 'True':
        # Instantiate Bert Classifier
        logging.info(' --- Testing model with Adapters ---')
        bert_classifier = BertTempProtClassifier(freeze_bert='True',adapter='True',mode='test').from_pretrained('./best_model_hugginface/model_hugginface')
    else:
        bert_classifier = BertTempProtClassifier(freeze_bert='True', mode='test').from_pretrained('./best_model_hugginface/model_hugginface')

    # Tell PyTorch to run the model on GPU
    bert_classifier = bert_classifier.to(device)

    # Load test Data
    val_data = pd.read_csv(val_data,header=None)

    logging.info('VAL SET distribution: {}'.format(Counter(val_data[2])))

    print('--Tokenizing val data...')
    # Run `preprocessing_for_bert` on the test set
    val_data[1]=[" ".join("".join(sample.split())) for sample in val_data[1]]
    val_inputs, val_masks = preprocessing_for_bert(val_data[1],MAX_LEN=512)

    # Label
    y_val=val_data[2]

    # Convert other data types to torch.Tensor
    val_labels = torch.tensor(y_val)

    print('--Creating ValDataloader...')
    # Create the DataLoader for our test set
    val_dataset = TensorDataset(val_inputs, val_masks,val_labels)
    val_sampler =  SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=val_batch_size,  num_workers = num_workers)

    print('--Start validation')

    # Compute predicted probabilities on the validation set
    probs = predict(bert_classifier, val_dataloader, device)

    if val_threshold:
        logging.info(f'Specified Threshold to use in the validation: {val_threshold}')
        evaluate_roc_valdata(probs, y_val, val_threshold)
    else: 
        logging.info('Threshold to be tuned')
        evaluate_roc_valdata(probs, y_val)


        