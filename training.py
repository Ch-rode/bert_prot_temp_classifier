from utilities import *
from model import *


def training_setup(train_data,lr,n_epochs,batch_size,max_len):
    ## 1. Create the checkpoint and best model directories
    ckp_dir = 'checkpoint'
    parent_dir = os.getcwd()
    best_dir = 'best_model'
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

    hugg_face="best_model_hugginface"

    hugg_face= os.path.join(parent_dir, hugg_face)

    try:
        os.mkdir(hugg_face) 
    except FileExistsError:
            pass

    
    ## 2.Model Summary

    logging.info('----- MODEL TRAINING PARAMETERS: ----- ')
    logging.info(' * TRAINING DATA {}'.format(train_data))
    logging.info(' * LEARNING RATE {}'.format(lr))
    logging.info(' * N EPOCHS:{}'.format(n_epochs))
    logging.info(' * BATCH SIZE {}'.format(batch_size))
    logging.info(' * MAX LEN {}'.format(max_len))


    ## 3. Set Random seed for reproducibility
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    return print("--Training configuration complete")


def data_prep(train_data,val_data,MAX_LEN,batch_size,num_workers):
    ### 3.1. Load Data
    print('--Starting Data Preparation and Tokenization')

    """The dataset is divided in 3 files, already splitted into training, validation, test set (created by using .py)"""
    logging.info('----- DATASET STRUCTURE ----- ')

    # Load train data
    train_data = pd.read_csv(train_data,header=None)
    # Display 5 samples from the test data
    print(train_data.sample(5))
    logging.info('TRAIN SET distribution:{}'.format(Counter(train_data[1])))

    # Load val data
    val_data = pd.read_csv(val_data,header=None)
    # Display 5 samples from the test data
    print(val_data.sample(5))
    logging.info('VAL SET distribution: {}'.format(Counter(val_data[1])))


    ## 4. Tokenization and DataPreparation
    logging.info('TOKENIZATION:')

    # Rostlab/prot_bert requires that the AA are separated between each other with a space
    train_data[0]= [" ".join("".join(sample.split())) for sample in train_data[0]]
    val_data[0]=[" ".join("".join(sample.split())) for sample in val_data[0]]

    #check
    print('Checking data prep: {}'.format(train_data.head()))


    # Run function `preprocessing_for_bert` on the train set, validation set and test set
    print('Tokenizing data...')
    train_inputs, train_masks = preprocessing_for_bert(train_data[0],MAX_LEN)
    val_inputs,val_masks = preprocessing_for_bert(val_data[0],MAX_LEN)

    #check
    logging.info("Example and check of the tokenized data: {}".format(train_data[0][0],train_inputs[0],train_masks[0]))


    ## 5 Create PyTorch DataLoader

    # Training label
    y_train=train_data[1]
    # Val label
    y_val=val_data[1]

    # Convert other data types to torch.Tensor
    train_labels = torch.tensor(y_train)
    val_labels = torch.tensor(y_val)

    print("Creating Train and Val Dataloader")
    # Create the DataLoader for our training set
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, num_workers = num_workers)

    # Create the DataLoader for our validation set
    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size, num_workers = num_workers)


    logging.info('Number of labels/target: {}'.format(len(np.unique(train_labels))))

    return(train_dataloader,val_dataloader,y_val)





def train(model, device, train_dataloader, val_dataloader, valid_loss_min_input, checkpoint_path, best_model_path, start_epochs, epochs, optimizer,scheduler,evaluation=True,adapter=None):
    """Train the BertClassifier model."""
    # Specify loss function
    loss_fn = nn.CrossEntropyLoss()
    device=device

    # Start training loop
    logging.info("--TRAINING\n")
    
    # Creating the config file of the model
    model.config.to_json_file("config.json")

    # Initialize tracker for minimum validation loss
    valid_loss_min = valid_loss_min_input 


    for epoch_i in range(start_epochs, epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        logging.info((f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}"))

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 500 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                logging.info(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        logging.info("-"*70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy = evaluate(model, val_dataloader,device)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            
            logging.info(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^10.6f} | {time_elapsed:^9.2f}")

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
    
        # saving the model in hugginface format
        model.save_pretrained('./best_model_hugginface/model_hugginface')

        if adapter == True:
            #save only the adapter separately 
            model.save_adapter('./best_model_hugginface/final_adapter','sequence_adapter')
    #model.save_pretrained("your-save-dir/) 3. After that you can load the model with Model.from_pretrained("your-save-dir/") 
    
    logging.info("-----------------Training complete--------------------------")

def evaluate(model, val_dataloader,device):
    """After the completion of each training epoch, measure the model's performance on our validation set."""
    loss_fn = nn.CrossEntropyLoss()
    # Put the model into the evaluation mode. The dropout layers are disabled during the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # Compute loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy








