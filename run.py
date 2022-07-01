from utils import *
from training import train, data_prep, training_setup
from model import initialize_model,BertTempProtConfig
from inference import run_inference,predict
from testing import test
from validation import validation


""" python init.py --mode inference --data_path  /home/rodella/NUOVI_DATI/TEST_cl_1  --best_model_path /home/rodella/NUOVI_DATI/Classifier/ratio1/best_model/best_model.pt  --device cuda  --pred_threshold 0.7 --num_workers 0 --max_len=512
"""

""" python init.py --mode train --train_data sampledata --val_data sampledata --test_data sampledata --n_epochs 2 --learning_rate 0.001 --max_len 512 --batch_size 4 --num_workers 0 --start_epochs 0 --train_first_run TRUE
"""

'''python init.py --mode test --test_data sampledata --num_workers 0 --device cuda --lr 0.001'''

def run(*argv):

    parser = argparse.ArgumentParser(description='BERT Classifier to discriminate between Mesophilic and Thermophilic sequences')

    
    parser.add_argument("--mode", help="Specify if you are in the 'train', 'inference', 'test' or 'validation' mode", type=str)

    # args common to all mode
    parser.add_argument("--num_workers", help="Specify if working with Mesophilic/Thermophilic", type=int)
    parser.add_argument("--max_len", help="Specify if working with Mesophilic/Thermophilic", type=int)
    parser.add_argument("--adapter", help="Specify if working with Mesophilic/Thermophilic", type=str)
    parser.add_argument("--fine_tuning", help="Specify if working with Mesophilic/Thermophilic", type=str)

    # training mode
    parser.add_argument("--train_data", help="Specify if working with Mesophilic/Thermophilic", type=str)
    parser.add_argument("--val_data", help="Specify if working with Mesophilic/Thermophilic", type=str)
    parser.add_argument("--test_data", help="Specify if working with Mesophilic/Thermophilic", type=str)
    parser.add_argument("--n_epochs", help="Specify if working with Mesophilic/Thermophilic", type=int)
    parser.add_argument("--learning_rate", help="Specify if working with Mesophilic/Thermophilic", type=float)
    parser.add_argument("--batch_size", help="Specify if working with Mesophilic/Thermophilic", type=int)
    parser.add_argument("--start_epochs", help="Specify if working with Mesophilic/Thermophilic", type=int)

    
    parser.add_argument("--train_first_run", help="Specify if working with Mesophilic/Thermophilic", type=str)
    

    # test mode
    parser.add_argument("--test_batch_size", help="Specify if working with Mesophilic/Thermophilic", type=int)
    #parser.add_argument("--val_threshold", help="Specify if working with Mesophilic/Thermophilic", type=float)


    # inference mode
    parser.add_argument("--data", help="Specify if working with Mesophilic/Thermophilic", type=str)
    parser.add_argument("--best_model_path", help="Specify if working with Mesophilic/Thermophilic", type=str)
    parser.add_argument("--pred_threshold", help="Specify if working with Mesophilic/Thermophilic", type=float)

    # validation mode
    parser.add_argument("--val_batch_size", help="Specify if working with Mesophilic/Thermophilic", type=int)
    parser.add_argument("--val_threshold", help="Specify if working with Mesophilic/Thermophilic", type=float)

    
            

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
        n_epochs = args.n_epochs # 10
        lr = args.learning_rate #5e-06
        threshold = args.pred_threshold #0.9
        batch_size = args.batch_size
        num_workers = args.num_workers
        start_epochs = args.start_epochs
        train_first_run = args.train_first_run
        fine_tuning=args.fine_tuning

        print('--Start Training setup')
        set_seed(42)    # Set seed for reproducibility
        training_setup(train_data,lr,n_epochs,batch_size,max_len)
        train_dataloader,val_dataloader,y_val= data_prep(train_data,val_data,MAX_LEN,batch_size,num_workers)

        print('All the model checkpoint are saved in the ./checkpoint/ and ./best_model/ folders.')

        if args.fine_tuning == 'True':
            fine_tuning = 'True'
            logging.info('Fine tuning: {}'.format(fine_tuning))
        else:
            fine_tuning = 'False'

        if args.adapter == 'True':
            bert_classifier, optimizer, scheduler = initialize_model(epochs=n_epochs,device=device,train_dataloader=train_dataloader,lr=lr,fine_tuning = 'False',adapter='True',mode='train') 
            
            if args.train_first_run:
                logging.info(' *** Training for first time ***')
                print('--Start Training first run')
                train(model = bert_classifier, optimizer=optimizer, scheduler=scheduler, train_dataloader = train_dataloader, val_dataloader = val_dataloader, start_epochs = start_epochs, epochs = n_epochs, valid_loss_min_input = np.Inf, evaluation = True,checkpoint_path = r"./checkpoint/current_checkpoint.pt", best_model_path = r"./best_model/best_model.pt",device=device,adapter='True')
            else:  
                logging.info(' *** Resume training from checkpoints ***')
                print('--Start Training from checkpoints')
                model, optimizer, epoch, valid_loss_min = load_ckp(r"./checkpoint/current_checkpoint.pt", bert_classifier, optimizer)
                logging.info('Epochs from checkpoint: {} '.format(epoch))
                logging.info("Check start_epoch for resume training= {}".format(start_epochs))
                logging.info("Valid_loss_min for starting epoch = {:.6f}".format(valid_loss_min))
                train(model = model, optimizer=optimizer, scheduler=scheduler, train_dataloader = train_dataloader, val_dataloader = val_dataloader, start_epochs = start_epochs, epochs = n_epochs, valid_loss_min_input = valid_loss_min, evaluation = True, checkpoint_path = r"./checkpoint/current_checkpoint.pt", best_model_path = r"./best_model/best_model.pt",device=device, adapter='True')

        else:
            bert_classifier, optimizer, scheduler = initialize_model(epochs=n_epochs,device=device,train_dataloader=train_dataloader,lr=lr,fine_tuning=fine_tuning) 
        
            if args.train_first_run:
                logging.info(' *** Training for first time ***')
                print('--Start Training first run')
                train(model = bert_classifier, optimizer=optimizer, scheduler=scheduler, train_dataloader = train_dataloader, val_dataloader = val_dataloader, start_epochs = start_epochs, epochs = n_epochs, valid_loss_min_input = np.Inf, evaluation = True,checkpoint_path = r"./checkpoint/current_checkpoint.pt", best_model_path = r"./best_model/best_model.pt",device=device)
            else: 
                logging.info(' *** Resume training from checkpoints *** ')
                print('--Start Training from checkpoints')
                model, optimizer, epoch, valid_loss_min = load_ckp(r"./checkpoint/current_checkpoint.pt", bert_classifier, optimizer)
                logging.info('Epochs from checkpoint: {} '.format(epoch))
                logging.info("Check start_epoch for resume training= {}".format(start_epochs))
                logging.info("Valid_loss_min for starting epoch = {:.6f}".format(valid_loss_min))
                train(model = model, optimizer=optimizer, scheduler=scheduler, train_dataloader = train_dataloader, val_dataloader = val_dataloader, start_epochs = start_epochs, epochs = n_epochs, valid_loss_min_input = valid_loss_min, evaluation = True, checkpoint_path = r"./checkpoint/current_checkpoint.pt", best_model_path = r"./best_model/best_model.pt",device=device)


        # Compute predicted probabilities on the validation set
        probs = predict(bert_classifier, val_dataloader,device)

        # Evaluate the Bert classifier and find the optimal threshold value using ROC
        logging.info('*** TUNING THRESHOLD USING ROC ON VAL DATA, ( updated until last bestmodel if training from checkpoints) ***')
        evaluate_roc_valdata(probs, y_val)


    if args.mode == 'val':

        val_data = args.val_data
        val_batch_size = args.val_batch_size
        num_workers = args.num_workers
        val_threshold=args.val_threshold # if want to use a desidered threshold otherwise it will tuned the best one
        adapter = args.adapter

        set_seed(42)    # Set seed for reproducibility
        if args.adapter == 'True' :
            if args.val_threshold:
                validation(val_data,val_batch_size, device, num_workers, adapter='True',val_threshold=val_threshold,)
            else:
                validation(val_data,val_batch_size, device, num_workers, adapter='True',val_threshold=None,)
        else: 
            if args.val_threshold:
                validation(val_data,val_batch_size, device, num_workers, adapter=None,val_threshold=val_threshold,)
            else:
                validation(val_data,val_batch_size, device, num_workers, adapter=None,val_threshold=None,)
        



    if args.mode == 'test':

        #print('--Removing checkpoints directory, keeping only best model')
        #shutil.rmtree(r"./best_model/") 
        print('--Start evaluation on test set')
        test_data = args.test_data
        lr = args.learning_rate #5e-06
        num_workers = args.num_workers
        adapter = args.adapter
        test_batch_size=args.test_batch_size
        mode=args.mode
        val_threshold=args.val_threshold
        adapter = args.adapter
        
        if adapter == 'True':
            test(test_data,num_workers,device,MAX_LEN,test_batch_size,val_threshold=val_threshold,adapter='True',mode='test')
        else:
            test(test_data,num_workers,device,MAX_LEN,test_batch_size,val_threshold=val_threshold,mode='test')

        


    if args.mode == 'inference':
        data=args.data
        best_model=args.best_model_path
        threshold=args.pred_threshold
        num_workers=args.num_workers


        print('--Starting Inference for the data: {}'.format(data))
        print('--Using the best model weights from: {}'.format(best_model))
        
        run_inference(data,best_model,device,threshold,num_workers,MAX_LEN)
        print('--Inference ended')
    


        

if __name__ == '__main__':
    import sys
    run(*sys.argv[1:])



















