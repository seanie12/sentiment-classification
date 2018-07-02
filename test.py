import tensorflow as tf
import numpy as np
import data_util
import os
from text_cnn import TextCNN
from config import Config
import csv

test_data_file = "data/test/tokenized_reviews.txt"
test_label_file = "data/test/labels.txt"
vocab_file = "data/vocab"
checkpoint_dir = "./save/checkpoints/cnn"
result_file = "./data/cnn_result.csv"
checkpoint_prefix = os.path.join(checkpoint_dir, "cnn")
max_vocab_size = 5e5
vocab = data_util.Vocab(vocab_file, max_vocab_size)
# load test data
test_docs, seq_len, max_len, test_labels = data_util.load_data(test_data_file,
                                                               test_label_file,
                                                               vocab)
config = Config(max_vocab_size, max_len)
model = TextCNN(config)
model.build()
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9
sess = tf.Session(config=sess_config)
init = tf.global_variables_initializer()
sess.run(init)
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt and tf.train.get_checkpoint_state(checkpoint_dir):
    model.restore(sess, ckpt.model_checkpoint_path)
else:
    print("no checkpoint saved")
    exit()
test_data = list(zip(test_docs, seq_len, test_labels))
batches = data_util.batch_loader(test_data, 32)
total_preds = []
for batch in batches:
    batch_docs, batch_seq_len, batch_labels = zip(*batch)
    preds = model.predict(sess, batch_docs, batch_seq_len)
    total_preds = np.concatenate([total_preds, preds], axis=0)
test_acc = np.mean(total_preds == test_labels)
rmse = np.mean(np.square(total_preds - test_labels), axis=0)
rmse = np.sqrt(rmse)
print("test accuracy : {}".format(test_acc))
print(rmse)
reviews = open(test_data_file, "r", encoding="utf-8").readlines()
reviews = list(map(lambda x : x.strip().replace("</s>", ""), reviews))
result = np.column_stack((total_preds, test_labels, reviews))
with open(result_file, "w", encoding="utf-8") as f:
    fw = csv.writer(f)
    fw.writerow(["prediction", "label", "review"])
    fw.writerows(result)
