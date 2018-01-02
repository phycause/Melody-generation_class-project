import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.contrib import legacy_seq2seq as seq2seq

flags = tf.flags
logging = tf.logging


flags.DEFINE_integer("state_size", 128,
                     "hidden layer size")
flags.DEFINE_integer("num_layers", 3,
                     "num_layers")
flags.DEFINE_integer("seq_length", 16,
                     "seq_length")
flags.DEFINE_integer("run_mode", 0,
                     "run_mode. train = 0 , generation = 1, valid = 2")

FLAGS = flags.FLAGS

class HParam():

    batch_size = 20
    n_epoch = 50
    learning_rate = 0.1
    decay_steps = 2000
    decay_rate = 0.9
    grad_clip = 5
    keep_prob = 0.35

    state_size = FLAGS.state_size
    num_layers = FLAGS.num_layers
    seq_length = FLAGS.seq_length
    log_dir = './logs' + '-ss=' + str(FLAGS.state_size) +'-nl=' + str(FLAGS.num_layers)+'-sl=' + str(FLAGS.seq_length)
    result_dir = ''
    metadata = 'metadata.tsv'
    gen_num = 500 # how many notes to generate


class DataGenerator():

    def __init__(self, datafiles, validfiles, is_valid, config):
        self.seq_length = config.seq_length
        self.batch_size = config.batch_size
        with open(datafiles, "r") as text_file:
            self.data = list(text_file.read().split(' '))

        self.is_valid = is_valid
        if is_valid:
            with open(validfiles, "r") as text_file:
                self.valid_data = list(map(int, text_file.read().split(' ')))
                

        self.total_len = len(self.data)  # total data length
        self.notes = list(set(self.data))
        self.notes.sort() #sort the notes
        
        self.note_pool_size = len(self.notes)  # Note Pool Size
        print('Note Pool Size: ', self.note_pool_size)
        self.note2id_dict = {w: i for i, w in enumerate(self.notes)} #i is index, w is note, use dictionary to save the indexes
        self.id2note_dict = {i: w for i, w in enumerate(self.notes)}

        # pointer position to generate current batch
        self._pointer = 0

        # save metadata file
        self.save_metadata(config.metadata)

    def note2id(self, c):
        return self.note2id_dict[c]

    def id2note(self, id):
        return self.id2note_dict[id]

    def save_metadata(self, file):
        with open(file, 'w') as f:
            f.write('id\tchar\n')
            for i in range(self.note_pool_size):
                c = self.id2note(i)
                f.write('{}\t{}\n'.format(i, c))

    def next_batch(self):
        self.data = self.valid_data if self.is_valid else self.data
        self.total_len = len(self.data)
        x_batches = []
        y_batches = []
        for i in range(self.batch_size):
            if self._pointer + self.seq_length + 1 >= self.total_len: #當pointer+sequence長度大於全部midi長度時，pointer歸零
                self._pointer = 0
            bx = self.data[self._pointer: self._pointer + self.seq_length]
            by = self.data[self._pointer +
                           1: self._pointer + self.seq_length + 1]
            self._pointer += self.seq_length  # update pointer position

            # convert to ids
            bx = [self.note2id(c) for c in bx]
            by = [self.note2id(c) for c in by]
            x_batches.append(bx)
            y_batches.append(by)

        return x_batches, y_batches


class Model():
    """
    The core LSTM recurrent neural network model.
    """

    def __init__(self, is_training, config, data, infer=False):
        if infer != 0: #如果要做valid或generation，要把batch_size, seq_length設成1
            config.batch_size = 1
            config.seq_length = 1
        with tf.name_scope('inputs'):
            self.input_data = tf.placeholder(
                tf.int32, [config.batch_size, config.seq_length])
            self.target_data = tf.placeholder(
                tf.int32, [config.batch_size, config.seq_length])

        with tf.name_scope('model'):
            self.cell = tf.contrib.rnn.LSTMBlockCell(config.state_size)
            self.cell = tf.contrib.rnn.MultiRNNCell([self.cell] * config.num_layers)
            self.initial_state = self.cell.zero_state(
                config.batch_size, tf.float32)
            with tf.variable_scope('rnnlm'):
                w = tf.get_variable(
                    'softmax_w', [config.state_size, data.note_pool_size])
                b = tf.get_variable('softmax_b', [data.note_pool_size])
                with tf.device("/cpu:0"):
                    embedding = tf.get_variable(
                        'embedding', [data.note_pool_size, config.state_size])
                    inputs = tf.nn.embedding_lookup(embedding, self.input_data)
                if is_training and config.keep_prob < 1:
                    inputs = tf.nn.dropout(inputs, config.keep_prob)
            outputs, last_state = tf.nn.dynamic_rnn(
                self.cell, inputs, initial_state=self.initial_state)

        with tf.name_scope('loss'):
            output = tf.reshape(outputs, [-1, config.state_size])

            self.logits = tf.matmul(output, w) + b
            self.probs = tf.nn.softmax(self.logits)
            self.last_state = last_state

            targets = tf.reshape(self.target_data, [-1])
            loss = seq2seq.sequence_loss_by_example([self.logits],
                                                    [targets],
                                                    [tf.ones_like(targets, dtype=tf.float32)])
            self.cost = tf.reduce_sum(loss) / config.batch_size
            tf.summary.scalar('loss', self.cost)

        with tf.name_scope('optimize'):
            self.lr = tf.placeholder(tf.float32, [])
            tf.summary.scalar('learning_rate', self.lr)

            optimizer = tf.train.AdamOptimizer()
            tvars = tf.trainable_variables()
            grads = tf.gradients(self.cost, tvars)
            for g in grads:
                tf.summary.histogram(g.name, g)
            grads, _ = tf.clip_by_global_norm(grads, config.grad_clip)

            self.train_op = optimizer.apply_gradients(zip(grads, tvars))
            self.merged_op = tf.summary.merge_all()


def train(data, is_valid, model, config):
    with tf.Session() as sess:
        train_loss_log = []
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(config.log_dir, sess.graph)

        max_iter = config.n_epoch * \
            (data.total_len // config.seq_length) // config.batch_size #計算iteration的數量，使用總數來計算
        
        for i in range(max_iter):
            # learning_rate = config.learning_rate * \
            #     (config.decay_rate ** (i // config.decay_steps))
            learning_rate = config.learning_rate
            x_batch, y_batch = data.next_batch()
            feed_dict = {model.input_data: x_batch,
                         model.target_data: y_batch, model.lr: learning_rate}
            train_loss, summary, _, _ = sess.run([model.cost, model.merged_op, model.last_state, model.train_op],
                                                 feed_dict)

            if i % 100 == 0:
                writer.add_summary(summary, global_step=i)                
                print('Step:{}/{}, training_loss:{:4f}'.format(i,
                                                               max_iter, train_loss))
                train_loss_log.append(train_loss)
                
            if i % 2000 == 0 or (i + 1) == max_iter:
                # print("learning rate: " + str(learning_rate))
                saver.save(sess, os.path.join(
                    config.log_dir, 'melody_gen_model.ckpt'), global_step=i)
                with open(config.log_dir + '/train_loss_log.txt', 'w') as text_file: #save train_loss_log
                    for idx in train_loss_log:
                        text_file.write(str(idx) + ',')


def sample(data, is_valid, model, config):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.latest_checkpoint(config.log_dir)
        # ckpt = os.path.join(config.log_dir, 'melody_gen_model.ckpt-14000')
        print(ckpt)
        saver.restore(sess, ckpt)

        # initial Value
        prime = ['62_0.13']
        if is_valid: #取出第一個data當input，第二個當target
            prime , ans= data.next_batch()
            prime = [data.id2note(prime[0][0])]

          
        
        state = sess.run(model.cell.zero_state(1, tf.float32))

        for note in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = data.note2id(note)
            feed = {model.input_data: x, model.initial_state: state}
            state = sess.run(model.last_state, feed)

        note = prime[-1]
        melody = prime
        valid_loss = 0
        for i in range(config.gen_num):
            x = np.zeros([1, 1])
            x[0, 0] = data.note2id(note)
            
            feed_dict = {model.input_data: x, model.initial_state: state}
            probs, state = sess.run([model.probs, model.last_state], feed_dict)

            if is_valid : #當要做valid的時候 加入valid需要的傳入值
                y = np.zeros([1, 1])
                y[0, 0] =ans[0][0]
                feed_dict_valid = {model.input_data: x, model.target_data: y, model.lr: 1}
                valid_loss = sess.run(model.cost, feed_dict_valid)
                print('valid loss:' + str(valid_loss))

            p = probs[0]
            note = data.id2note(np.argmax(p))
            print(note)
            sys.stdout.flush()
            time.sleep(0.05)
            melody.append(note)
        with open(config.log_dir + '/melody.txt', 'w') as text_file: #save the melody it generated
            for idx in melody:
                text_file.write(str(idx) + ' ')
        return melody


def main(infer):

    config = HParam()
    is_training = True if infer == 0 else False #判斷是否為training，用在dropout
    is_valid = True if infer == 2 else False

    train_data_path = 'data/melodygen.train.txt'
    valid_data_path = ''
    data = DataGenerator(train_data_path, valid_data_path, is_valid, config)

    model = Model(is_training, config, data, infer=infer)

    run_fn = train if infer==0 else sample #選擇是sample還是用train，如果是sample是1

    run_fn(data, is_valid, model, config) #


if __name__ == '__main__':
    msg = """
    Usage:
    Training: 
        python melody_gen_new.py 0
    Generating:
        python melody_gen_new.py 1
    Validation:
        python melody_gen_new.py 2
    """

    infer = FLAGS.run_mode
    print('--Training--' if infer == 0 else '')
    print('--Generating--' if infer == 1 else '')
    print('--Validation--' if infer == 2 else '')

    main(infer)

