
from utilities import *
from model import *
from training import *
from inference import predict



def evaluate_test(model, val_dataloader,device):
    """After the completion of each training epoch, measure the model's performance on our validation set."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss()


    val_loss = []
    all_logits = []

    # For each batch in our test set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
        #print(b_labels)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
            #print(logits)
        all_logits.append(logits)

        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

    all_logits = torch.cat(all_logits, dim=0)
    print(all_logits)

    probs = F.softmax(all_logits, dim=1).cpu().numpy()
    print(probs)

    #from tensor([[-0.9970,  0.9971]]) to array([[0.11982378, 0.8801762 ]], dtype=float32
    val_loss = np.mean(val_loss)

    return probs, val_loss


def test(test_data,num_workers,device,MAX_LEN,lr,test_batch_size,val_threshold,adapter=None,mode=None,):
    """Evaluate the Bert classifier on Test set"""
    logging.info('--Evaluation on the TEST SET')


    if adapter == 'True':
        # Instantiate Bert Classifier
        logging.info(' --- Testing model with Adapters ---')
        bert_classifier = BertTempProtClassifier(freeze_bert='True',adapter='True',mode=mode).from_pretrained('./best_model_hugginface/model_hugginface')
    else:
        bert_classifier = BertTempProtClassifier(freeze_bert='True').from_pretrained('./best_model_hugginface/model_hugginface')

    # Tell PyTorch to run the model on GPU
    bert_classifier = bert_classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=lr,    # Default learning rate
                      eps=1e-8    # Default epsilon value
                      )

    # Load test Data
    test_data = pd.read_csv(test_data,header=None)
    # Display 5 samples from the test data
    print(test_data.sample(5))
    logging.info('TEST SET distribution: {}'.format(Counter(test_data[1])))

    print('Tokenizing test data...')
    # Run `preprocessing_for_bert` on the test set
    test_data[0]=[" ".join("".join(sample.split())) for sample in test_data[0]]
    test_inputs, test_masks = preprocessing_for_bert(test_data[0],MAX_LEN)

    # Label
    y_test=test_data[1]

    # Convert other data types to torch.Tensor
    test_labels = torch.tensor(y_test)

    print('Creating TestDataloader...')
    # Create the DataLoader for our test set
    test_dataset = TensorDataset(test_inputs, test_masks,test_labels)
    test_sampler =  SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=test_batch_size,  num_workers = num_workers)

    print('--Start evaluation')
    all_probs, val_loss = evaluate_test(bert_classifier, test_dataloader,device)

    logging.info("Evaluation on the test set ended, test loss: {:.4f} ".format(val_loss))


    # Evaluate the Bert classifier and find the optimal threshold value using ROC
    logging.info('--ROC AND METRICS ON TEST DATA, on best model')
    evaluate_roc_testdata(all_probs, y_test,val_threshold)

    return True

