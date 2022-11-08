# Intro

This module aims to train the sentence representation model which is based on the constractive learning architecture ([SIMCSE](https://github.com/princeton-nlp/SimCSE)).

# Train

To train such a model, firstly you should prepare the triplet dataset. Please refers to the

```
data_builder.py
```

Please put the data into

```
data/
```

folder.

Then, you need to download the [original SIMCSE model](https://huggingface.co/princeton-nlp/unsup-simcse-roberta-base) in the

```
model/simcse_sup_roberta
```

folder.

Next, please run the following command to train the model:

```
# python train.py \
    --model_name_or_path '../model/simcse_sup_roberta' \
    --train_file '../data/train.csv' \
    --output_dir '../../result/' \
    --num_train_epochs 3 \
    --per_device_train_batch_size 64 \
    --learning_rate 5e-5 \
    --max_seq_length 64 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"

```

Please follow the [readme file](https://github.com/princeton-nlp/SimCSE) of SIMCSE for further details.

The result model is stored in the

```
'../../result/'
```

folder.

# Model

You can directly reuse our tuned model [here](https://drive.google.com/file/d/1-aVwmWzWQL1n8RMwsCwqErpz3UJe2Plm/view?usp=sharing)
