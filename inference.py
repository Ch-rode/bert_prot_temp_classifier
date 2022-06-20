from utilities import *
from model import  preprocessing_for_bert, BertTempProtClassifier, tokenizer



#pu<<<
def predict(model, dataloader,device):
    """Perform a forward pass on the trained BERT model to predict probabilities on a set."""
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    all_logits = []

    # For each batch in our test set...
    for batch in dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
            #print(logits)
        all_logits.append(logits)
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)
    #print(all_logits)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()
    #print(probs)

    return probs

def run_inference(data,best_model,device,threshold,num_workers,max_len):
    """Perform full inference classification on a set of data"""
    set_seed(42)    # Set seed for reproducibility

    # using pytorch checkpoints
    #checkpoint=torch.load(best_model)
    #model=BertTempProtClassifier()
    #model.load_state_dict(checkpoint['state_dict'])

    device=device
    MAX_LEN=max_len
 
    # using hugginface model saved
    model=BertTempProtClassifier.from_pretrained(best_model)
    model = model.to(device)


    model.eval()

    #MAX_LEN=512

    # Load the data to inference from
    data=pd.read_csv(data,header=None)
    data=[" ".join("".join(sample.split())) for sample in data[0]]
    print("Data Check:", data[0])
    inputs, masks = preprocessing_for_bert(data,MAX_LEN)

    #inputs = inputs.to(device)
    #masks = masks.to(device)


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


    for index, item in enumerate(data):
        #print(data[i],preds[i])
        print("Sequence to Classify: {} Class predicted: {}...".format(item[:5],preds[index]))
        logging.info("Sequence to Classify: {} Class predicted: {}...".format(item[:5],preds[index]))
    
    return preds

