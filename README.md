# classifier_python
ADAPTER TRAINIG FIRST RUN

python run.py --mode train --train_data TEST_cl_1 --val_data  TEST_cl_1  --n_epochs 2 --learning_rate 5e-06 --max_len 512 --batch_size 2  --num_workers 0 --start_epochs 0 --train_first_run TRUE  --device cuda --fine_tuning False --adapter True

ADAPTER TRAINING FROM CHECKPOINTS (RICORDARSI DI PARTIRE DA START EPOCHS NEL SENSO ULTIMA EPOCA FATTA)

python run.py 
--mode train 
--train_data TEST_cl_1
--val_data  TEST_cl_1 
--start_epochs 2
--n_epochs 4 
--learning_rate 5e-06 
--max_len 512 
--batch_size 2  
--num_workers 0  
--device cuda 

--fine_tuning False 
--adapter True



ADAPTER VALIDATION using own threshold
python run.py --mode val --val_threshold 0.2012 --val_data TEST_cl_1 --val_batch_size 2 --adapter True --num_workers 0


ADAPTER VALIDATION tuning the threshold 
python run.py --mode val --val_data TEST_cl_1 --val_batch_size 2 --adapter True --num_workers 0


ADAPTER TEST VEDERE SE HANNO RISOLTO BUG
python run.py --mode test --val_threshold 0.2021 --test_data TEST_cl_1 --test_batch_size 2 --adapter True --num_workers 0


ADAPTER INFERENCE  


------------------------------------------------------------------

FULL TUNING TRAINING
python run.py --mode train --train_data TEST_cl_1 --val_data  TEST_cl_1  --n_epochs 2 --learning_rate 5e-06 --max_len 512 --batch_size 2  --num_workers 0 --start_epochs 0 --train_first_run TRUE  --device cuda --fine_tuning True 

FULL TUNING WITH OUR THRESHOLD
python run.py --mode val --val_threshold 0.2012 --val_data TEST_cl_1 --val_batch_size 2  --num_workers 0 --fine_tuning True 



FULL TUNING TUNING THE THRESHOLD
python run.py --mode val --val_threshold 0.2012 --val_data TEST_cl_1 --val_batch_size 2  --num_workers 0 --fine_tuning True 



FULL TUNING TEST

python run.py --mode test --val_threshold 0.2021 --test_data TEST_cl_1 --test_batch_size 2  --num_workers 0 --fine_tuning True 

FULL TUNING INFERENCE

python run.py --mode inference --pred_threshold 0.2021 --data inference_data  --num_workers 0 --best_model_path ./best_model_hugginface/model_hugginface


