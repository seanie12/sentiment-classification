import tensorflow as tf
from tensorflow.contrib import layers


class TextCNN(object):
    def __init__(self, config):
        self.config = config
        self.input_x = None
        self.input_y = None
        self.sequence_length = None
        self.dropout = None
        self.embedding_matrix = None
        self.feature_maps = None
        self.logits = None
        self.preds = None
        self.accuracy = None
        self.rmse = None
        self.loss = None
        self.train_op = None
        self.global_step = tf.Variable(tf.constant(0), trainable=False,
                                       name="global_step")

    def add_placeholder(self):
        self.input_x = tf.placeholder(shape=[None, self.config.max_length],
                                      dtype=tf.int32, name="input_x")
        self.input_y = tf.placeholder(shape=[None], dtype=tf.int32)
        self.dropout = tf.placeholder(dtype=tf.float32)
        self.sequence_length = tf.placeholder(shape=[None], dtype=tf.int32,
                                              name="seq_len")

    def add_embedding(self):
        zero = tf.constant([[0.0] * self.config.embedding_size],
                           dtype=tf.float32)
        embedding_matrix = tf.get_variable(
            shape=[self.config.vocab_size - 1, self.config.embedding_size],
            dtype=tf.float32,
            initializer=layers.xavier_initializer(), name="embedding")
        self.embedding_matrix = tf.concat([zero, embedding_matrix], axis=0)

    def add_conv(self):
        pooled = []
        lookup = tf.nn.embedding_lookup(self.embedding_matrix, self.input_x)
        expanded_words = tf.expand_dims(lookup, axis=3)
        for i, filter_size in enumerate(self.config.filters):
            # for wide convolution, zero pad left and right with filter - 1
            paddings = tf.constant(
                [[0, 0], [filter_size - 1, filter_size - 1], [0, 0], [0, 0]])
            padded_chars = tf.pad(expanded_words, paddings, mode="CONSTANT")
            filter_shape = [filter_size, self.config.embedding_size, 1,
                            self.config.num_filters]
            w = tf.get_variable(shape=filter_shape,
                                initializer=layers.xavier_initializer(),
                                name="filter_{}".format(i), dtype=tf.float32)
            b = tf.get_variable(shape=self.config.num_filters,
                                initializer=tf.zeros_initializer(),
                                name="bias_{}".format(i), dtype=tf.float32)
            conv = tf.nn.conv2d(padded_chars, w, padding="VALID",
                                strides=[1, 1, 1, 1])
            conv = tf.nn.relu(conv + b)
            max_pool = tf.nn.max_pool(conv, ksize=[1,
                                                   self.config.max_length - filter_size + 1,
                                                   1, 1], padding="VALID",
                                      strides=[1, 1, 1, 1])
            pooled.append(max_pool)
        self.feature_maps = tf.concat(pooled, axis=3)
        self.feature_maps = tf.nn.dropout(self.feature_maps, self.dropout)
        self.feature_maps = tf.reshape(self.feature_maps,
                                       [-1, self.config.num_filters * len(
                                           self.config.filters)])

    def add_logits(self):
        self.logits = layers.fully_connected(self.feature_maps,
                                             self.config.num_classes,
                                             activation_fn=None)
        self.preds = tf.argmax(self.logits, axis=1, output_type=tf.int32)
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(self.input_y, self.preds), dtype=tf.float32))
        mse = tf.reduce_mean((self.input_y - self.preds) ** 2)
        self.rmse = tf.sqrt(tf.cast(mse,tf.float32))

    def add_loss(self):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.input_y)
        self.loss = tf.reduce_mean(losses)

    def add_train_op(self):
        opt = tf.train.AdamOptimizer(self.config.lr)
        self.train_op = opt.minimize(self.loss, global_step=self.global_step)

    def build(self):
        self.add_placeholder()
        self.add_embedding()
        self.add_conv()
        self.add_logits()
        self.add_loss()
        self.add_train_op()

    def train(self, sess, docs, sequence_lengths, labels, dropout):
        input_feed = {
            self.input_x: docs,
            self.sequence_length: sequence_lengths,
            self.input_y: labels,
            self.dropout: dropout
        }
        output_feed = [self.train_op, self.global_step, self.loss,
                       self.accuracy, self.rmse]
        outputs = sess.run(output_feed, feed_dict=input_feed)
        # return global step, loss and accuracy
        return outputs[1], outputs[2], outputs[3], outputs[4]

    def eval(self, sess, docs, seq_len, labels, dropout=1.0):
        input_feed = {
            self.input_x: docs,
            self.input_y: labels,
            self.sequence_length: seq_len,
            self.dropout: dropout
        }
        output_feed = [self.accuracy, self.rmse]
        outputs = sess.run(output_feed, input_feed)
        # return accuracy and rmse
        return outputs[0], outputs[1]

    def predict(self, sess, docs, sequence_lengths, dropout=1.0):
        feed_dict = {
            self.input_x: docs,
            self.sequence_length: sequence_lengths,
            self.dropout: dropout

        }
        preds = sess.run(self.preds, feed_dict=feed_dict)
        return preds

    def save(self, sess, path, global_step):
        saver = tf.train.Saver(tf.global_variables())
        save_path = saver.save(sess, path, global_step)
        print("save session to {}".format(save_path))

    def restore(self, sess, path):
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, path)
        print("load model from {}".format(path))
