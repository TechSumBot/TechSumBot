from keras_bert import get_pretrained, PretrainedList, get_checkpoint_paths
from keras_bert import load_trained_model_from_checkpoint, Tokenizer

model_path = '/workspace/src/_1_module/model/asnq_tanda_checkpoint/models/tanda_bert_base_asnq/'
paths = get_checkpoint_paths(model_path)
checkpoint_path = model_path+'model.ckpt'

# print(paths.config, paths.checkpoint, paths.vocab)
bert_model = load_trained_model_from_checkpoint(paths.config, checkpoint_path, seq_len=None)
