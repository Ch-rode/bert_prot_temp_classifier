from utils import *
from training import train, data_prep, training_setup
from model import initialize_model
from inference import run_inference, predict
from testing import test
from validation import validation





def run(*argv):

    parser = argparse.ArgumentParser(description='BERT Classifier to discriminate between Mesophilic and Thermophilic sequences')

    
    parser.add_argument("--mode", help="Specify if you are in the 'train', 'inference', 'test' or 'validation' mode", type=str)

    # args common to all mode
    parser.add_argument("--num_workers", help="The number of processes to use", type=int, default=0)
    parser.add_argument("--max_len", help="The maximum input sequence length, default 512. Sequences longer than this will be truncated", type=int,default=512)
    parser.add_argument("--adapter", help="Insert 'True' if want to work with an Adapter Model form adapter-hub", type=str,default=None)
    parser.add_argument("--fine_tuning", help="Insert 'True' if want to train the model as a full-finetuning, 'False' will be tune only the classification head", type=str)

    # training mode
    parser.add_argument("--train_data", help="A csv/txt containing the training data in the format ID,SEQUENCE,LABEL encoded as mesophilic=0 and thermophilic=1", type=str)
    parser.add_argument("--val_data", help="A csv/txt containing the validation data in the format ID,SEQUENCE,LABEL encoded as mesophilic=0 and thermophilic=1", type=str)
    parser.add_argument("--n_epochs", help="Total number of epochs to run the training", type=int)
    parser.add_argument("--learning_rate", help="Initial learning rate", type=float)
    parser.add_argument("--batch_size", help="Dataloader batch size for the specific mode", type=int)
    parser.add_argument("--start_epochs", help="Starting epoch, default 0. If resume from checkpoint insert the last one fulled trained as starting epoch", type=int,default=0)
    parser.add_argument("--resume_from_checkpoint", help="If the training should continue from the /checkpoint/current_checkpoint.pt folder.", type=str,default=None)
    

    # test mode
    #--batch_size
    #--threshold
    parser.add_argument("--test_data", help="A csv/txt containing the testing data in the format ID,SEQUENCE,LABEL encoded as mesophilic=0 and thermophilic=1", type=str)
    parser.add_argument("--best_model_path", 
                        help="Path to pretrained model or model identifier from huggingface.co/models. Specify it if not working in the same directory as trainig, othrwise it will use the default one after training", 
                        type=str,default='./best_model_hugginface/model_hugginface')

    


    # inference mode
    parser.add_argument("--inference_data", help="File containing sequences to infere from line by line", type=str)
    #--best_model_path
    parser.add_argument("--threshold", help="Threshold to apply for the validation, test or inference mode. In the validation mode if left blank it will tune the threshold finding the best one", type=float)

    # validation mode
    #--batch_size
    #--threshold"
    #--best_model_path
    
            

    ## Set up GPU if available
    logging.info('----- GPU INFORMATION: ----- ')
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        logging.info(f'There are {str(torch.cuda.device_count())} GPU(s) available.')
        for i in range(torch.cuda.device_count()):
            logging.info(f'GPU Device name: {torch.cuda.get_device_name(int(i))}, ')
    else:
        logging.info('No GPU available, using the CPU instead.')
        device = torch.device("cpu")


    args = parser.parse_args()

    max_len = args.max_len #512
    if args.max_len :
        MAX_LEN=max_len
    else:
        MAX_LEN=512
    

    if args.mode == 'train':
        train_data = args.train_data
        val_data = args.val_data
        n_epochs = args.n_epochs 
        lr = args.learning_rate #5e-06
        batch_size = args.batch_size
        num_workers = args.num_workers
        start_epochs = args.start_epochs
        #resume_from_checkpoint= args.resume_from_checkpoint
        fine_tuning=args.fine_tuning

        print('--Start Training preprocessing')
        set_seed(42)    # Set seed for reproducibility
        training_setup(train_data,lr,n_epochs,batch_size,max_len)
        train_dataloader,val_dataloader,y_val= data_prep(train_data,val_data,MAX_LEN,batch_size,num_workers)
        print('--End Training setup')

        print('All the model checkpoint are saved in the ./checkpoint/ and ./best_model/ folders.')

        if args.fine_tuning == 'True':
            fine_tuning = 'True'
            logging.info('Fine tuning: {}'.format(fine_tuning))
        else:
            fine_tuning = 'False'

        if args.adapter == 'True':
            bert_classifier, optimizer, scheduler = initialize_model(epochs=n_epochs,device=device,train_dataloader=train_dataloader,lr=lr,adapter='True',mode='train') 
            
            if args.resume_from_checkpoint:
                logging.info(' *** Resume training from checkpoints ***')
                print('--Start Training from checkpoints from ./checkpoint/current_checkpoint.pt')
                model, optimizer, epoch, valid_loss_min = load_ckp(r"./checkpoint/current_checkpoint.pt", bert_classifier, optimizer)
                logging.info('Epochs from checkpoint: {} '.format(epoch))
                logging.info("Check start_epoch for resume training= {}".format(start_epochs))
                logging.info("Valid_loss_min for starting epoch = {:.6f}".format(valid_loss_min))
                train(model = model, optimizer=optimizer, scheduler=scheduler, train_dataloader = train_dataloader, val_dataloader = val_dataloader, start_epochs = start_epochs, epochs = n_epochs, valid_loss_min_input = valid_loss_min, evaluation = True, checkpoint_path = r"./checkpoint/current_checkpoint.pt", best_model_path = r"./best_model/best_model.pt",device=device, adapter='True')
            else:  
                logging.info(' *** Training for first time ***')
                print('--Start Training first run')
                train(model = bert_classifier, optimizer=optimizer, scheduler=scheduler, train_dataloader = train_dataloader, val_dataloader = val_dataloader, start_epochs = start_epochs, epochs = n_epochs, valid_loss_min_input = np.Inf, evaluation = True,checkpoint_path = r"./checkpoint/current_checkpoint.pt", best_model_path = r"./best_model/best_model.pt",device=device,adapter='True')
                

        else:
            bert_classifier, optimizer, scheduler = initialize_model(epochs=n_epochs,device=device,train_dataloader=train_dataloader,lr=lr,fine_tuning=fine_tuning,mode='train') 
        
            if args.resume_from_checkpoint:
                logging.info(' *** Resume training from checkpoints *** ')
                print('--Start Training from checkpoints from ./checkpoint/current_checkpoint.pt')
                model, optimizer, epoch, valid_loss_min = load_ckp(r"./checkpoint/current_checkpoint.pt", bert_classifier, optimizer)
                logging.info('Epochs from checkpoint: {} '.format(epoch))
                logging.info("Check start_epoch for resume training= {}".format(start_epochs))
                logging.info("Valid_loss_min for starting epoch = {:.6f}".format(valid_loss_min))
                train(model = model, optimizer=optimizer, scheduler=scheduler, train_dataloader = train_dataloader, val_dataloader = val_dataloader, start_epochs = start_epochs, epochs = n_epochs, valid_loss_min_input = valid_loss_min, evaluation = True, checkpoint_path = r"./checkpoint/current_checkpoint.pt", best_model_path = r"./best_model/best_model.pt",device=device)

            else: 
                logging.info(' *** Training for first time ***')
                print('--Start Training first run')
                train(model = bert_classifier, optimizer=optimizer, scheduler=scheduler, train_dataloader = train_dataloader, val_dataloader = val_dataloader, start_epochs = start_epochs, epochs = n_epochs, valid_loss_min_input = np.Inf, evaluation = True,checkpoint_path = r"./checkpoint/current_checkpoint.pt", best_model_path = r"./best_model/best_model.pt",device=device)
                

        # Compute predicted probabilities on the validation set
        probs = predict(bert_classifier, val_dataloader,device)

        # Evaluate the Bert classifier and find the optimal threshold value using ROC
        logging.info('*** TUNING THRESHOLD USING ROC ON VAL DATA, ( updated until last bestmodel if training from checkpoints) ***')
        evaluate_roc_valdata(probs, y_val)


    if args.mode == 'val':

        val_data = args.val_data
        val_batch_size = args.batch_size
        num_workers = args.num_workers
        val_threshold=args.threshold # if want to use a desidered threshold otherwise it will tuned the best one
        adapter = args.adapter
        best_model = args.best_model_path

        set_seed(42)    # Set seed for reproducibility
        if args.adapter == 'True' :
            if args.val_threshold:
                validation(val_data,val_batch_size, device, num_workers, best_model,adapter='True',val_threshold=val_threshold)
            else:
                validation(val_data,val_batch_size, device, num_workers, best_model,adapter='True',val_threshold=None,)
        else: 
            if args.val_threshold:
                validation(val_data,val_batch_size, device, num_workers, best_model, val_threshold=val_threshold,)
            else:
                validation(val_data,val_batch_size, device, num_workers, best_model,val_threshold=None,)
        



    if args.mode == 'test':

        #print('--Removing checkpoints directory, keeping only best model')
        #shutil.rmtree(r"./best_model/") 
        print('--Start evaluation on test set')
        test_data = args.test_data
        #lr = args.learning_rate #5e-06
        num_workers = args.num_workers
        adapter = args.adapter
        test_batch_size=args.batch_size
        mode=args.mode
        val_threshold=args.threshold
        best_model=args.best_model_path
        
        
        if adapter == 'True':
            test(test_data,num_workers,device,MAX_LEN,test_batch_size,best_model,val_threshold=val_threshold,adapter='True',mode='test')
        else:
            test(test_data,num_workers,device,MAX_LEN,test_batch_size,best_model, val_threshold=val_threshold,mode='test')

        print('--End evaluation on test set')


    if args.mode == 'inference':
        data=args.inference_data
        best_model=args.best_model_path
        threshold=args.threshold
        num_workers=args.num_workers
        adapter=args.adapter

        data=pd.read_csv(data,header=None) 
        logging.info('** INFERENCE ** ')
        print('--Starting Inference for the data: {}'.format(data))
        print('--Using the best model weights from: {}'.format(best_model))
        
        if args.adapter:
            logging.info('Running inference with adapter model')
            run_inference(data,best_model,device,threshold,num_workers,MAX_LEN,adapter='True')
        else:
            logging.info('Running inference')
            run_inference(data,best_model,device,threshold,num_workers,MAX_LEN)
        
        print('--Inference ended')
    


        

if __name__ == '__main__':
    import sys
    run(*sys.argv[1:])



















