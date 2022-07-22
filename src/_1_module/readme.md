# Intro
Module I is the usefulness ranking module. It lies on the BERT-base classification model which is trained on ASQN dataset. 
# Training
To train such a model, please use the hugginface transformer library in the folder:
```
transformers
```
The training script is 
```
CUDA_VISIBLE_DEVICES=1 python run_glue.py
--model_type bert
--model_name_or_path bert-base-uncased
--task_name asqn
--do_train     
--do_lower_case     
--data_dir ../data/asqn/    
 --per_gpu_train_batch_size 16     
 --learning_rate 1e-5     
 --num_train_epochs 3     
 --output_dir ../../model/module_1
```
The training and validation data can be downloaded from [the origial github repo of TANDA paper](https://github.com/alexa/wqa_tanda)
# Models
