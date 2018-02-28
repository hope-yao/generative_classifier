
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils.utils import *
slim = tf.contrib.slim

class Orig_Classifier():
    def __init__(self):
        self.batch_size = 32
        self.imsize = 28
        self.channels = 1
        self.num_class = 10
        self.learning_rate = 0.01
        self.mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
        self.training_epochs = 200
        self.logdir, self.modeldir = creat_dir('MNIST')

        self.input = tf.placeholder(tf.float32, [self.batch_size, self.imsize, self.imsize, self.channels])
        self.labels = tf.placeholder(tf.float32, [self.batch_size, self.num_class])
        # self.input = tf.Variable(self.input_pl)

    def add_layers(self, images, num_classes=10, is_training=False,
              dropout_keep_prob=0.5,
              prediction_fn=slim.softmax,
              scope='LeNet'):
        end_points = {}

        net = slim.conv2d(images, 32, 5, scope='conv1')
        net = slim.max_pool2d(net, 2, 2, scope='pool1')
        net = slim.conv2d(net, 64, 5, scope='conv2')
        net = slim.max_pool2d(net, 2, 2, scope='pool2')
        net = slim.flatten(net)
        end_points['Flatten'] = net

        mid_output = net = slim.fully_connected(net, 1024, scope='fc3')
        # net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
        #                    scope='dropout3')
        logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                      scope='fc4')

        end_points['mid_output'] = mid_output
        end_points['Logits'] = logits
        end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

        return end_points

    def build_model(self, input, reuse=False):

        with tf.variable_scope('MNIST', reuse=reuse) as mnist_var:
            self.end_points = self.add_layers(input, num_classes=10, is_training=True,
                           dropout_keep_prob=0.99)
            self.mnist_var = tf.contrib.framework.get_variables(mnist_var)

            # Define loss and optimizer, minimize the squared error
            self.rec_err = tf.reduce_mean(tf.abs(self.end_points['Predictions'] - self.labels))
            self.rec_cost = self.imsize * self.imsize * tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.end_points['Logits']))

            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            grads_g = optimizer.compute_gradients(self.rec_cost, var_list=self.mnist_var)
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

        total_batch = int(self.mnist.train.num_examples/self.batch_size)
        # Training cycle
        for epoch in range(self.training_epochs):
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, batch_ys = self.mnist.train.next_batch(self.batch_size)
                feed_dict_train = {self.input: batch_xs.reshape(self.batch_size, self.imsize, self.imsize, 1), self.labels:batch_ys}#[:,0:1]
                _, rec_cost_val, rec_err_val = self.sess.run([self.apply_gradient_training, self.rec_cost, self.rec_err], feed_dict_train)

            # Display logs per epoch step
            batch_xs, batch_ys = self.mnist.test.next_batch(self.batch_size)
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
