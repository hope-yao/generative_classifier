
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils.utils import *
slim = tf.contrib.slim
from data.data_loader import smallnorb

class Orig_Classifier():
    def __init__(self):
        self.batch_size = 32
        self.imsize = 96
        self.channels = 1
        self.num_class = 5
        self.learning_rate = 0.001
        # self.mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
        self.training_epochs = 200
        self.logdir, self.modeldir = creat_dir('smallNORB')

        self.input = tf.placeholder(tf.float32, [self.batch_size, self.imsize, self.imsize, self.channels])
        self.labels = tf.placeholder(tf.float32, [self.batch_size, self.num_class])
        # self.input = tf.Variable(self.input_pl)

    def add_layers(self, images, num_classes=10, is_training=False,
              dropout_keep_prob=0.5,
              prediction_fn=slim.softmax):

        end_points = {}

        net1 = slim.conv2d(images, 32, 5, scope='conv1')
        net1_1 = slim.conv2d(net1, 32, 5, scope='conv1_1')
        net2 = slim.max_pool2d(images+net1_1, 2, 2, scope='pool1')
        net3 = slim.conv2d(net2, 64, 5, scope='conv2')
        net3_1 = slim.conv2d(net3, 64, 5, scope='conv2_1')
        net3_2 = slim.conv2d(net3_1, 64, 5, scope='conv2_2')
        net4 = slim.max_pool2d(net3+net3_2, 2, 2, scope='pool2')
        net5 = slim.conv2d(net4, 128, 5, scope='conv3')
        net5_1 = slim.conv2d(net5, 128, 5, scope='conv3_1')
        net5_2 = slim.conv2d(net5_1, 128, 5, scope='conv3_2')
        net6 = slim.max_pool2d(net5+net5_2, 2, 2, scope='pool3')
        net7 = slim.conv2d(net6, 256, 5, scope='conv4')
        net7_1 = slim.conv2d(net7, 256, 5, scope='conv4_1')
        net7_2 = slim.conv2d(net7_1, 256, 5, scope='conv4_2')
        net8 = slim.max_pool2d(net7+net7_2, 2, 2, scope='pool4')
        # net9 = slim.conv2d(net8, 512, 5, scope='conv5')
        # net9_1 = slim.conv2d(net9, 512, 5, scope='conv5_1')
        # net9_2 = slim.conv2d(net9_1, 512, 5, scope='conv5_2')
        # net10 = slim.max_pool2d(net9_2, 2, 2, scope='pool5')
        net11 = slim.flatten(net8)

        mid_output0 = net12 = slim.fully_connected(net11, 512, scope='fc3')
        mid_output1 = net13 = slim.fully_connected(net12, 64, scope='fc4')
        mid_output2 = net14 = slim.fully_connected(net13, 5, activation_fn=None, scope='fc5')
        # net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
        #                    scope='dropout3')
        logits = slim.fully_connected(net13, num_classes, activation_fn=None, scope='fc6')

        # end_points['last_conv'] = net10
        end_points['Flatten'] = net11
        end_points['mid_output0'] = mid_output0
        end_points['mid_output1'] = mid_output1
        end_points['mid_output2'] = mid_output2
        end_points['Logits'] = logits
        end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

        return end_points

    def build_model(self, input, reuse=False):

        with tf.variable_scope('Norb', reuse=reuse) as net_var:
            self.end_points = self.add_layers(input, num_classes=self.num_class, is_training=True,
                           dropout_keep_prob=0.99)
            self.net_var = tf.contrib.framework.get_variables(net_var)

            # Define loss and optimizer, minimize the squared error
            self.rec_err = tf.reduce_mean(tf.abs(self.end_points['Predictions'] - self.labels))
            # self.rec_err = tf.metrics.accuracy(labels=self.labels, predictions=self.end_points['Predictions'])
            self.rec_cost = self.imsize * self.imsize * tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.end_points['Logits']))

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        grads_g = optimizer.compute_gradients(self.rec_cost, var_list=self.net_var)
        self.apply_gradient_training = optimizer.apply_gradients(grads_g)

        return self.end_points

    def train_model(self):
        # Launch the graph
        FLAGS = tf.app.flags.FLAGS
        tfconfig = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True,
        )
        tfconfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tfconfig)
        # Initializing the variables
        self.sess.run(tf.initialize_variables(set(tf.all_variables())))#-set([self.input])))

        data_dir = '/home/exx/Documents/Hope/generative_classifier/data/'
        train_img, train_label, test_img, test_label = smallnorb(data_dir)
        total_batch = int(len(train_img)/self.batch_size)
        # Training cycle
        for epoch in range(self.training_epochs):
            # Loop over all batches
            for i in range(total_batch):
                batch_xs = train_img[i*self.batch_size:(i+1)*self.batch_size]
                batch_ys = train_label[i*self.batch_size:(i+1)*self.batch_size]
                feed_dict_train = {self.input: batch_xs.reshape(self.batch_size, self.imsize, self.imsize, 1), self.labels:batch_ys}#[:,0:1]
                _, rec_cost_val, rec_err_val = self.sess.run([self.apply_gradient_training, self.rec_cost, self.rec_err], feed_dict_train)
            # self.sess.run(tf.reduce_mean(tf.abs(tf.gradients(self.rec_cost, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Norb/fc4/weights:0")[0])[0])), feed_dict_train)
            # Display logs per epoch step
            batch_xs = test_img[i * self.batch_size:(i + 1) * self.batch_size]
            batch_ys = test_label[i * self.batch_size:(i + 1) * self.batch_size]
            feed_dict_test = {self.input: batch_xs.reshape(self.batch_size, self.imsize, self.imsize, 1), self.labels:batch_ys}
            test_rec_cost_val, test_rec_err_val = self.sess.run([self.rec_cost, self.rec_err], feed_dict_test)
            print("Epoch:", '%04d' % (epoch + 1),
                  "train_rec_cost=", "{:.9f}".format(rec_cost_val),
                  "test_rec_cost=", "{:.9f}".format(test_rec_cost_val),
                  "train_rec_err=", "{:.9f}".format(rec_err_val),
                  "test_rec_err=", "{:.9f}".format(test_rec_err_val),
                  )

            if epoch % 50 == 0:
                self.saver = tf.train.Saver()
                snapshot_name = "%s_%s" % ('experiment', str(epoch))
                fn = self.saver.save(self.sess, "%s/%s.ckpt" % (self.modeldir, snapshot_name))
                print("Model saved in file: %s" % fn)

        print("Optimization Finished!")



if __name__ == '__main__':
    classifier = Orig_Classifier()
    classifier.build_model(classifier.input)
    classifier.train_model()


