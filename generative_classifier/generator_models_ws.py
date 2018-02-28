import tensorflow as tf
slim = tf.contrib.slim


def mysample(mean_var, num_net, is_vae):
    '''    samlping process for VAE with subnetworks also computes the KL divergence

    :param mean_var:
    :param num_net:
    :param is_vae:
    :return:
    '''
    kld_sub = []
    sample_sub = []
    batch_size, latent_dim = mean_var.get_shape().as_list()
    latent_dim /= num_net

    for i in range(num_net):
        mean_var_i = tf.slice(mean_var, (0, latent_dim*i), (batch_size, latent_dim))
        if is_vae:
            z_mean, z_log_var = tf.split(mean_var_i, 2, axis=1)
            epsilon = tf.random_normal(shape=z_mean.get_shape(), mean=0., name='epsilon')
            sample_i = z_mean + tf.exp(z_log_var / 2) * epsilon
            sample_sub += [sample_i]
            kld_i = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
            kld_sub += [kld_i]
        else:
            sample_sub += [mean_var_i]
    return sample_sub, kld_sub


def mnist28_encoder(x, latent_dim, reuse=False):
    '''
    encoder for mnist digits
    :param x:
    :param latent_dim:
    :return:
    '''
    batch_size, _, _, _ = x.get_shape().as_list()
    # Encoder Hidden layer with sigmoid activation #1
    with tf.variable_scope('Enc', reuse=reuse):
        h1 = slim.conv2d(x, 16, 3, 2, activation_fn=tf.nn.relu)
        h2 = slim.conv2d(h1, 32, 3, 2, activation_fn=tf.nn.relu)
        h3 = slim.conv2d(h2, 64, 3, 2, activation_fn=tf.nn.relu)
        h4 = slim.fully_connected(tf.reshape(h3, (batch_size, 64 * 4 * 4)), latent_dim,
                                  activation_fn=None)
    return h4


def mnist28_decoder(sample_sub, label, reuse=False):
    '''
    decoder with subnetworks for mnist digits
    :param sample_sub:
    :param label:
    :return:
    '''
    imsize = 28
    batch_size, _ = sample_sub[0].get_shape().as_list()
    out_sub = []
    for i, sample_i in enumerate(sample_sub):
        with tf.variable_scope('Dec{}'.format(i), reuse=reuse):
            h1 = slim.fully_connected(sample_i, 64 * 4 * 4, activation_fn=tf.nn.relu)
            h2 = slim.conv2d_transpose(tf.reshape(h1, (batch_size, 4, 4, 64)), 64, 3, 2, activation_fn=tf.nn.relu)
            h3 = slim.conv2d_transpose(h2, 32, 3, 2, activation_fn=tf.nn.relu)
            h4 = slim.conv2d_transpose(h3, 16, 3, 2, activation_fn=tf.nn.relu)
            out = slim.conv2d(h4, 1, 5, 1, activation_fn=None, padding='valid')

        out_sub += [out]
        label_i = tf.tile(tf.reshape(label[:, i:i + 1], (batch_size, 1, 1, 1)),
                          (1, imsize, imsize, 1))
        out_sum = out * label_i if i == 0 else out_sum + out * label_i
        # out_sum = out
    return out_sum, out_sub



def mnist36_encoder(self, x):
    '''
    encoder for overlapping mnist digits
    :param self:
    :param x:
    :return:
    '''
    with tf.variable_scope('Enc'):
        h1 = slim.conv2d(x, 16, 3, 2, activation_fn=tf.nn.relu)
        h2 = slim.conv2d(h1, 32, 3, 2, activation_fn=tf.nn.relu)
        h3 = slim.conv2d(h2, 64, 3, 2, activation_fn=tf.nn.relu)
        h4 = slim.fully_connected(tf.reshape(h3, (self.batch_size, 64 * 5 * 5)), 2 * self.num_net * self.latent_dim,
                                  activation_fn=None)
    return h4


def mnist36_decoder(self, sample_sub, label):
    '''
    decoder with subnetworks for overlapping mnist digits
    :param self:
    :param sample_sub:
    :param label:
    :return:
    '''
    out_sub = []
    for i, sample_i in enumerate(sample_sub):
        with tf.variable_scope('Dec{}'.format(i)):
            h1 = slim.fully_connected(sample_i, 64 * 5 * 5, activation_fn=tf.nn.relu)
            h2 = slim.conv2d_transpose(tf.reshape(h1, (self.batch_size, 5, 5, 64)), 64, 3, 2, activation_fn=tf.nn.relu)
            h3 = slim.conv2d_transpose(h2, 32, 3, 2, activation_fn=tf.nn.relu)
            h4 = slim.conv2d_transpose(h3, 16, 3, 2, activation_fn=tf.nn.relu)
            out = slim.conv2d(h4, 1, 5, 1, activation_fn=None, padding='valid')

            out_sub += [out]
            label_i = tf.tile(tf.reshape(label[:, i:i + 1], (self.batch_size, 1, 1, 1)),
                              (1, self.imsize, self.imsize, 1))
            out_sum = out * label_i if i == 0 else out_sum + out * label_i
            # out_sum = out
    return out_sum, out_sub