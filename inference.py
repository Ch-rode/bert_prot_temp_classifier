from utils import *
from model import  preprocessing_for_bert, BertTempProtClassifier, tokenizer, BertTempProtAdapterClassifier


def predict(model, dataloader,device):
    """Perform a forward pass on the trained BERT model to predict probabilities on a set."""
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    set_seed(42)
    model.eval()

    all_logits = []

    # For each batch in our test set...
    for batch in dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    return probs

def run_inference(data,best_model,device,threshold,num_workers,max_len,adapter=None):
    """Perform full inference classification on a set of data"""

    set_seed(42)    # Set seed for reproducibility
    
    logging.info((f'Uploading the best_model model from {best_model}'))
    logging.info((f'Data to infere from {data}'))
    
    # using pytorch checkpoints
    #checkpoint=torch.load(best_model)
    #model=BertTempProtClassifier()
    #model.load_state_dict(checkpoint['state_dict'])

    device=device
    MAX_LEN=max_len
 
    # using hugginface model saved
    if adapter == 'True':
        model = BertTempProtAdapterClassifier.from_pretrained(best_model)
    else:
        model = BertTempProtClassifier.from_pretrained(best_model)

    
    model = model.to(device)


    model.eval()

    # Load the data to inference from (must be a df in this point)
    #the format of the file is ID, SEQUENCE 
    #data=pd.read_csv(data,header=None) 
    data=[" ".join("".join(sample.split())) for sample in data[0]]
    print("Data Check:", data[0])
    inputs, masks = preprocessing_for_bert(data,MAX_LEN)


    # Create the DataLoader for the data
    dataset = TensorDataset(inputs, masks)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=32,  num_workers = num_workers)

    

    ###  Predictions

    # Compute predicted probabilities on the  set
    print('Bert Prediction inference...')
    probs = predict(model, dataloader,device)

    # Get predictions from the probabilities
    threshold = threshold
    preds = np.where(probs[:, 1] > threshold, 1, 0)

    mappings = {0: 'Mesophilic', 1: 'Thermophilic'}

    preds=[mappings[x] for x in preds] 



    f = open("classification.out","w")



    for index, item in enumerate(data):
        #print(data[i],preds[i])
        print("Sequence to Classify: {} Class predicted: {}...".format(item[:5],preds[index]))
        #logging.info("Sequence to Classify: {} Class predicted: {}...".format(item[:5],preds[index]))
        item2=item.replace(" ", "")
        f.write(str(item2 + ',' + str(probs[index]) + ',' + preds[index] +'\n'))

        print('All classification results saved in the classification.out file')
    
    return preds

