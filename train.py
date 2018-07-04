import tensorflow as tf
import data_util
from config import Config
from rnn import TextRNN
import numpy as np
import os

train_data_file = "data/train/tokenized_reviews.txt"
train_label_file = "data/train/labels.txt"
dev_data_file = "data/dev/tokenized_reviews.txt"
dev_label_file = "data/dev/labels.txt"
vocab_file = "data/vocab"
checkpoint_dir = "./save/checkpoints/rnn"
checkpoint_prefix = os.path.join(checkpoint_dir, "cnn")
max_vocab_size = 5e5
batch_size = 32
print("build vocab")
# data_util.Vocab.build_vocab_file(train_data_file, vocab_file)
vocab = data_util.Vocab(vocab_file, max_vocab_size=max_vocab_size)
# load data
print("load data file")
train_docs, train_seq_len, max_len, train_labels = \
    data_util.load_data(train_data_file, train_label_file, vocab)
print(max_len)
dev_docs, dev_seq_len, max_len, dev_labels = \
    data_util.load_data(dev_data_file, dev_label_file, vocab, max_len)

# set config, and build tf graph and init
config = Config(max_vocab_size, max_len)
model = TextRNN(config)
model.build()
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9
sess = tf.Session(config=sess_config)
init = tf.global_variables_initializer()
sess.run(init)
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    print("reload model")
    model.restore(sess, ckpt.model_checkpoint_path)
else:
    print("start from scratch")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

print("start training")
best_acc = 0
for i in range(config.num_epochs):
    epoch = i + 1
    train_data = list(zip(train_docs, train_seq_len, train_labels))
    batches = data_util.batch_loader(train_data, batch_size, shuffle=True)
    for batch in batches:
        batch_docs, batch_seq_len, batch_labels = zip(*batch)
        step, loss, acc, rmse = model.train(sess, batch_docs, batch_seq_len,
                                            batch_labels, config.dropout)
        print("epoch: {}, loss: {}, acc: {}".format(epoch, loss, acc))
    del train_data
    dev_data = list(zip(dev_docs, dev_seq_len, dev_labels))
    dev_batches = data_util.batch_loader(dev_data, batch_size)
    total_preds = []
    for dev_batch in dev_batches:
        batch_docs, batch_seq_len, batch_labels = zip(*dev_batch)
        preds = model.predict(sess, batch_docs, batch_seq_len, dropout=1.0)
        total_preds = np.concatenate([total_preds, preds], axis=0)
    dev_acc = np.mean(total_preds == dev_labels)
    print("dev accuracy : {}".format(dev_acc))
    if dev_acc > best_acc:
        print("new record : {}".format(dev_acc))
        best_acc = dev_acc
        model.save(sess, checkpoint_prefix, model.global_step.eval(sess))
