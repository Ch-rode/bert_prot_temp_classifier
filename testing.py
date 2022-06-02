
from utilities import *
from model import *
from training import *


def test(test_data,num_workers,device,MAX_LEN,lr,adapter=None):
    """Evaluate the Bert classifier on Test set"""
    logging.info('--Evaluation on the TEST SET')

    if adapter == True:
        # Instantiate Bert Classifier
        logging.info(' --- Training with Adapters ---')
        bert_classifier = BertClassifierAdapter(freeze_bert=False)
    else:
        bert_classifier = BertClassifier(freeze_bert=False)

    # Tell PyTorch to run the model on GPU
    bert_classifier = bert_classifier.to(device)
    #bert_classifier =  DataParallelPassthrough(bert_classifier).to(device)
    #bert_classifier = nn.DataParallel(bert_classifier).to(device) # multi - gpu batch size distributed training

    # Create the optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=lr,    # Default learning rate
                      eps=1e-8    # Default epsilon value
                      )
    
    model, optimizer, start_epoch, valid_loss_min = load_ckp(r"./best_model/best_model.pt", bert_classifier, optimizer)

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
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=32,  num_workers = num_workers)


    val_loss, val_accuracy = evaluate(model, test_dataloader,device)

    logging.info("Evaluation on the test set ended, val loss: {:.4f}, val acc: {:.2f}%".format(val_loss, val_accuracy))

    return print("--Evaluation on the test set ended, val loss: {:.4f}, val acc: {:.2f}%".format(val_loss, val_accuracy))


