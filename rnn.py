import tensorflow as tf
import tensorflow.contrib.layers as layers


class TextRNN(object):
    def __init__(self, config):
        self.config = config
        self.input_x = None
        self.input_y = None
        self.sequence_lengths = None
        self.dropout = None
        self.last_states = None
        self.embedding_matrix = None
        self.logits = None
        self.loss = None
        self.train_op = None
        self.preds = None
        self.accuracy = None
        self.rmse = None
        self.global_step = tf.Variable(tf.constant(0), trainable=False,
                                       name="global_step")

    def add_placeholder(self):
        self.input_x = tf.placeholder(shape=[None, self.config.max_length],
                                      dtype=tf.int32, name="input_x")
        self.input_y = tf.placeholder(shape=[None], dtype=tf.int32,
                                      name="label")
        self.dropout = tf.placeholder(dtype=tf.float32, name="dropout")
        self.sequence_lengths = tf.placeholder(shape=[None], dtype=tf.int32,
                                               name="seq_len")

    def add_embeddings(self):
        with tf.variable_scope("embedding"):
            zeros = tf.constant([[0.0] * self.config.embedding_size],
                                dtype=tf.float32)
            embedding_matrix = tf.get_variable(
                shape=[self.config.vocab_size - 1, self.config.embedding_size],
                initializer=layers.xavier_initializer(), name="embedding")
            self.embedding_matrix = tf.concat([zeros, embedding_matrix], axis=0)

    def add_rnn(self):
        embedded_chars = tf.nn.embedding_lookup(self.embedding_matrix,
                                                self.input_x)
        cell_fw = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size)
        cell_bw = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size)
        ((_, _),
         (fw_states, bw_states)) = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                   cell_bw,
                                                                   embedded_chars,
                                                                   self.sequence_lengths,
                                                                   dtype=tf.float32)
        # state is tuple of (cell state, hidden_state)
        # get last hidden state : [batch, hidden_size]
        last_states = tf.concat([fw_states[-1], bw_states[-1]], axis=1)
        self.last_states = tf.nn.dropout(last_states, self.dropout)

    def add_logits(self):
        self.logits = layers.fully_connected(self.last_states,
                                             self.config.num_classes,
                                             activation_fn=None)

    def add_loss(self):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.input_y, logits=self.logits)
        self.loss = tf.reduce_mean(loss)
        self.preds = tf.argmax(self.logits, axis=1, output_type=tf.int32)
        mse = tf.reduce_mean((self.input_y - self.preds) ** 2)
        self.rmse = tf.sqrt(tf.cast(mse, tf.float32))
        correct_pred = tf.cast(tf.equal(self.preds, self.input_y), tf.float32)
        self.accuracy = tf.reduce_mean(correct_pred)

    def add_train_op(self):
        opt = tf.train.AdamOptimizer(self.config.lr)
        # gradient clipping
        tvars = tf.trainable_variables()
        grads = tf.gradients(self.loss, tvars)
        clipped_grads, norm = tf.clip_by_global_norm(grads, 5.0)
        self.train_op = opt.apply_gradients(zip(clipped_grads, tvars),
                                            global_step=self.global_step)

    def build(self):
        self.add_placeholder()
        self.add_embeddings()
        self.add_rnn()
        self.add_logits()
        self.add_loss()
        self.add_train_op()

    def save(self, sess, path):
        saver = tf.train.Saver(tf.global_variables())
        save_path = saver.save(sess, path)
        print("save the model at {}".format(save_path))

    def restore(self, sess, path):
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, path)
        print("reload the model from {}".format(path))

    def train(self, sess, docs, seq_len, labels, dropout):
        feed_dict = {
            self.input_x: docs,
            self.input_y: labels,
            self.sequence_lengths: seq_len,
            self.dropout: dropout
        }
        output_feed = [self.train_op, self.global_step, self.loss,
                       self.accuracy, self.rmse]
        outputs = sess.run(output_feed, feed_dict)
        return outputs[1], outputs[2], outputs[3], outputs[4]

    def eval(self, sess, docs, seq_len, labels, dropout=1.0):
        feed_dict = {
            self.input_x: docs,
            self.input_y: labels,
            self.sequence_lengths: seq_len,
            self.dropout: dropout
        }
        output_feed = [self.accuracy, self.rmse]
        outputs = sess.run(output_feed, feed_dict)
        return outputs[0], outputs[1]

    def predict(self, sess, docs, seq_len, dropout=1.0):
        feed_dict = {
            self.input_x: docs,
            self.sequence_lengths: seq_len,
            self.dropout: dropout
        }
        output_feed = [self.preds]
        outputs = sess.run(output_feed, feed_dict)
        return outputs[0]
