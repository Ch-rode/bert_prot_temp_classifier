# BertTempProtClassifier


[Hugginface Model Repository](https://huggingface.co/Chrode/bert_prot_temp_classifier) 
In the Hugginface-Hub the model is a only-classification-head tuned model. 


Demo on Colab (Running on Chrome): [Colab](https://colab.research.google.com/drive/19OPKXZJTO2ofd6_agS1nAj9-OSYZVz87?usp=sharing) 



* If training a model with **adapters** add the following argument indipendently from the selected mode: `--adapter True`, from [Adapter-Hub library](https://adapterhub.ml/) ![image](https://user-images.githubusercontent.com/61243245/177858361-b3d6b33e-f675-4eb0-834e-96c3933ed776.png)

* If training a model with **full-fine tuning** add for all the mode: `--fine_tuning True`


* If training a model with fine-tuning only the **classification head**: `--fine_tuning False`


###  TRAINING

```
python run.py 
--mode train
--train_data TRAIN_cl_1 
--val_data VAL_cl_1 
--start_epochs 0 
--n_epochs 4 
--batch_size 16 
--learning_rate 5e-06 
--num_workers 0
--max_len 512 
```
If resume from checkpoint add the following arguments: `--resume_from_checkpoint True`. It will load the current checkpoints from the default directory.
Remember that `--start_epochs` is the last epoch totally processed.

### VALIDATION 

```
python run.py
--val_data VAL_cl_1
--batch_size 8
--num_workers 0
--best_model Chrode/bert_prot_temp_classifier #from hugginface or by default the trained saved model './best_model_hugginface/model_hugginface'
```

If you want to do the validation with a desidered **threshold** add i.e. `--threshold 0.3691`, otherwise if you don't put the argument the code will find the best threshold tuned on validation data added.



### TESTING

```
python run.py
--mode test 
--threshold 0.3691 
--test_data TEST_cl_1 
--batch_size 8 
--num_workers 0
--best_model_path Chrode/bert_prot_temp_classifier
```

### INFERENCE 

```
python run.py
--inference_data data
--threshold 0.3691
--best_model_path Chrode/bert_prot_temp_classifier
--num_workers 0 
```



