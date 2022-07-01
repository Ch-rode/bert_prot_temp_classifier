
from utils import *
from model import *
from training import *
from inference import predict



def evaluate_test(model, test_dataloader, device):
    """After the completion of each training epoch, measure the model's performance on our validation set."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss()


    test_loss = []
    all_logits = []
    batch_loss = 0 

    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
            all_logits.append(logits)

        loss = loss_fn(logits, b_labels)

        batch_loss += loss.item()
        test_loss.append(loss.item())

    all_logits = torch.cat(all_logits, dim=0)
    print(all_logits)

    probs = F.softmax(all_logits, dim=1).cpu().numpy()
    print(probs)
    #from tensor([[-0.9970,  0.9971]]) to array([[0.11982378, 0.8801762 ]], dtype=float32

    test_loss = np.mean(test_loss)
    avg_test_loss = batch_loss / len(test_dataloader)
    logging.info("Check test loss: {:.4f} ".format(avg_test_loss))

    return probs, test_loss 


def test(test_data,num_workers,device,MAX_LEN,test_batch_size,val_threshold,adapter=None,mode=None,):
    """Evaluate the Bert classifier on Test set"""
    logging.info(' ** Evaluation on the TEST SET **')


    if adapter == 'True':
        # Instantiate Bert Classifier
        logging.info(' --- Testing model with Adapters ---')
        bert_classifier = BertTempProtClassifier(freeze_bert='True',adapter='True',mode='test').from_pretrained('./best_model_hugginface/model_hugginface')
    else:
        bert_classifier = BertTempProtClassifier(freeze_bert='True',mode='test').from_pretrained('./best_model_hugginface/model_hugginface')

    # Tell PyTorch to run the model on GPU
    bert_classifier = bert_classifier.to(device)


    # Load test Data
    # the format of the test file is ID, SEQUENCE, LABEL
    test_data = pd.read_csv(test_data,header=None,sep=',')
    import copy
    sequences = copy.copy(test_data[1])

    # Display 5 samples from the test data
    print(test_data.sample(5))
    logging.info('TEST SET distribution: {}'.format(Counter(test_data[2])))

    print('--Tokenizing test data...')
    # Run `preprocessing_for_bert` on the test set
    test_data[1]=[" ".join("".join(sample.split())) for sample in test_data[1]]
    test_inputs, test_masks = preprocessing_for_bert(test_data[1],MAX_LEN)

    # Label
    y_test=test_data[2]

    # Convert other data types to torch.Tensor
    test_labels = torch.tensor(y_test)

    print('--Creating TestDataloader...')
    # Create the DataLoader for our test set
    test_dataset = TensorDataset(test_inputs, test_masks,test_labels)
    test_sampler =  SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=test_batch_size,  num_workers = num_workers)

    print('--Start evaluation')
    all_probs, test_loss = evaluate_test(bert_classifier, test_dataloader,device)

    logging.info("Evaluation on the test set ended, test loss: {:.4f} ".format(test_loss))
    
    
    rows = zip(test_data[0], sequences, test_data[2], all_probs)
    import csv

    with open('test.out', "w") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

    # Evaluate the Bert classifier and find the optimal threshold value using ROC
    logging.info('ROC AND METRICS ON TEST DATA, by best model')
    evaluate_roc_testdata(all_probs, y_test,val_threshold)

    print('--End evaluation')

    return True

