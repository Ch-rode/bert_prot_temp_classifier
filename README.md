# classifier_python

If training for first time a model (not resume from checkpoints) add the following arguments:
```
--train_first_run TRUE  
```
Don't specify anything if you resume from checkpoints, only remember that `--start_epochs` is the last epoch totally processed.

If training a model with adapters add for all the mode:

```
--adapter True
--fine_tuning False
```

If training a model with full-fine tuning add for all the mode:
```
--fine_tuning True
```


If training a model with fine-tuning only the classification head:
```
--fine_tuning False
```


###  TRAINING

```
python run.py 
--mode train 
--train_data TEST_cl_1 
--val_data  TEST_cl_1  
--n_epochs 4
--learning_rate 5e-06 
--max_len 512 
--batch_size 16  
--num_workers 0 
--start_epochs 0 
```

### VALIDATION tuning the threshold 

```
python run.py
--val_data data
--mode val 
--val_data TEST_cl_1 
--val_batch_size 8
--num_workers 0
```

### VALIDATION using specified threshold

```
python run.py
--mode val 
--val_threshold 0.2012 
--val_data TEST_cl_1 
--val_batch_size 8 
--num_workers 0

```



### TESTING

```
python run.py
--mode test 
--val_threshold 0.2021 
--test_data TEST_cl_1 
--test_batch_size 8 
--num_workers 0

```

### INFERENCE  


