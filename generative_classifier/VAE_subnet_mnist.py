
""" Auto Encoder Example.
Using an auto encoder on MNIST handwritten digits.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf

slim = tf.contrib.slim
from utils.utils import *
from tensorflow.examples.tutorials.mnist import input_data
from generative_classifier.generator_models import *
from utils.plot_figures import *


class VAE_subnet(object):
    def __init__(self):
        # Launch the graph
        FLAGS = tf.app.flags.FLAGS
        tfconfig = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True,
        )
        tfconfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tfconfig)

        self.is_vae = True

        # Parameters
        self.learning_rate = 0.0001
        self.training_epochs = 200
        self.batch_size = 32
        self.display_step = 1
        self.examples_to_show = 10
        self.num_net = 10
        self.latent_dim = 8

        # Network Parameters
        self.imsize = 28# MNIST data input (img shape: 28*28)

        self.logdir, self.modeldir = creat_dir('VAE_sbunet')
        # copyfile('./generative_classifier/VAE_subnet_mnist.py', self.modeldir + '/' + 'VAE_subnet_mnist.py')
        # copyfile('./generative_classifier/generator_models.py', self.modeldir + '/' + 'generator_models.py')
        # copyfile('./utils/plot_figures.py', self.modeldir + '/' + 'plot_figures.py')
        # copyfile('./adversarial_attacks/generate_FGSM_attacks.py', self.modeldir + '/' + 'generate_FGSM_attacks.py')

        # tf Graph input (only pictures)
        self.X = tf.placeholder("float", [self.batch_size, self.imsize, self.imsize, 1])
        self.label = tf.placeholder("float", [self.batch_size, self.num_net])

        self.encoder = mnist28_encoder
        self.decoder = mnist28_decoder

        self.mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


    def build_training_model(self):
        # Construct model
        self.mean_var = self.encoder(self.X, self.num_net*self.latent_dim)

        self.sample_sub, self.kld_sub = mysample(self.mean_var, self.num_net, is_vae=self.is_vae)

        self.out_sum, self.out_sub = self.decoder(self.sample_sub, self.label)
        self.rec = tf.sigmoid(self.out_sum)

        # Define loss and optimizer, minimize the squared error
        self.rec_err = tf.reduce_mean(tf.abs(self.rec-self.X))
        self.rec_cost = self.imsize*self.imsize*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.X, logits=self.out_sum))
        if self.is_vae:
            self.kl_mean = tf.reduce_mean(list2tensor(self.kld_sub))
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.rec_cost + self.kl_mean)
        else:
            self.kl_mean = tf.constant(0.)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.rec_cost)

        self.summary_writer = tf.summary.FileWriter(self.logdir)
        self.summary_op_train = tf.summary.merge([
            tf.summary.scalar("train/kl_mean_train", self.kl_mean),
            tf.summary.scalar("train/rec_cost_train", self.rec_cost),
            tf.summary.scalar("train/rec_err", self.rec_err),
            tf.summary.scalar("train/lr", self.learning_rate),
            # tf.summary.scalar("grad/grad1", grad1),
            # tf.summary.scalar("grad/grad2", grad2),
            # tf.summary.scalar("grad/grad3", grad3),
            # tf.summary.scalar("grad/grad4", grad4),
            # tf.summary.scalar("grad/grad5", grad5),
            # tf.summary.scalar("grad/grad6", grad6),
        ])

        self.summary_op_test = tf.summary.merge([
            tf.summary.scalar("test/kl_mean_test", self.kl_mean),
            tf.summary.scalar("test/rec_cost_test", self.rec_cost),
            tf.summary.scalar("test/rec_err", self.rec_err),
        ])



    def optimizing(self):
        # Initializing the variables
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        total_batch = int(self.mnist.train.num_examples/self.batch_size)

        # Training cycle
        for epoch in range(self.training_epochs):
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, batch_ys = self.mnist.train.next_batch(self.batch_size)
                feed_dict_train = {self.X: batch_xs.reshape(self.batch_size, self.imsize, self.imsize, 1), self.label:batch_ys}#[:,0:1]
                _, rec_cost_val, kld_val = self.sess.run([self.optimizer, self.rec_cost, self.kl_mean], feed_dict_train)

            # Display logs per epoch step
            if epoch % self.display_step == 0:
                batch_xs, batch_ys = self.mnist.test.next_batch(self.batch_size)
                feed_dict_test = {self.X: batch_xs.reshape(self.batch_size, self.imsize, self.imsize, 1), self.label:batch_ys}
                test_rec_cost_val, test_kld_val = self.sess.run([self.rec_cost, self.kl_mean], feed_dict_test)
                print("Epoch:", '%04d' % (epoch+1),
                      "rec_cost_end=", "{:.9f}".format(rec_cost_val),
                      "test_rec_cost_end=", "{:.9f}".format(test_rec_cost_val),
                      )
                summary_train = self.sess.run(self.summary_op_train, feed_dict_train)
                summary_test = self.sess.run(self.summary_op_test, feed_dict_test)
                self.summary_writer.add_summary(summary_train, epoch)
                self.summary_writer.add_summary(summary_test, epoch)
                self.summary_writer.flush()

            if epoch % 50 == 0:
                self.saver = tf.train.Saver()
                snapshot_name = "%s_%s" % ('experiment', str(epoch))
                fn = self.saver.save(self.sess, "%s/%s.ckpt" % (self.modeldir, snapshot_name))
                print("Model saved in file: %s" % fn)

        print("Optimization Finished!")


if __name__ == "__main__":

    with tf.variable_scope('VAE_subnet', reuse=False):
        with tf.device('/device:GPU:3'):
            vaesub = VAE_subnet()
            vaesub.build_training_model()
            vaesub.optimizing()

    feed_dict = {
        vaesub.X: vaesub.mnist.test.images[:vaesub.batch_size].reshape(vaesub.batch_size, vaesub.imsize, vaesub.imsize, 1),
        vaesub.label: vaesub.mnist.test.labels[:vaesub.batch_size]}
    visualize_generator_performance(vaesub, feed_dict, vaesub.modeldir+'/test.png')
    print('done')
