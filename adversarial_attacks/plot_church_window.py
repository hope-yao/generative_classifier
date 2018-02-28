import tensorflow as tf
import numpy as np
from original_classifier.orig_classifier_mnist import Orig_Classifier
from adversarial_attacks.generate_FGSM_attacks import get_attack_directions_lenet
from utils.plot_figures import *
from generative_classifier.VAE_subnet_mnist import VAE_subnet


def get_church_window_lenet(cfg, lenet, sess, orig_img, delta_vec):
    resolution = cfg['resolution']
    axis_range = cfg['range']
    pred_matrix = np.zeros((resolution, resolution))
    for i, w_x in enumerate(np.linspace(axis_range[0], axis_range[1], resolution)):
        for j, w_y in enumerate(np.linspace(axis_range[2], axis_range[3], resolution)):
            test_img = orig_img + w_x * delta_vec['x'] + w_y * delta_vec['y']
            feed_dict = {lenet.input_pl_var: test_img.reshape(1, 28, 28, 1)}
            test_pred = sess.run(lenet.end_points['Predictions'], feed_dict)[0]
            pred_matrix[i, j] = np.argmax(test_pred)
    # transpose becasue matrix and image different axis notation
    return  pred_matrix



def get_church_window_generator(cfg, model, sess, orig_img, orig_label, delta_vec):
    from generative_classifier.generative_classification import fast_build_classify_model, fast_testing_classify_model

    para_list = [0.3, 0.3, 0.6, 0.005, 5.0, -2.0]

    with tf.variable_scope('VAE_subnet'):

        resolution = cfg['resolution']
        pred_matrix = []
        rec_err = []
        axis_range = cfg['range']

        gen_classify_hist = []
        test_img_batch = []
        apply_gradient_op, apply_gradient_op_refine, init_op = fast_build_classify_model(model)
        for i,w_x in enumerate(np.linspace(axis_range[0], axis_range[1], resolution)):
            for j,w_y in enumerate(np.linspace(axis_range[2], axis_range[3], resolution)):
                test_img_batch += [orig_img + w_x * delta_vec['x'] + w_y * delta_vec['y']]
                if len(test_img_batch) == model.batch_size:
                    imgs = np.asarray(test_img_batch).reshape(model.batch_size,model.imsize,model.imsize,1)
                    labels = np.tile(orig_label, (model.batch_size,1))
                    out_dict = fast_testing_classify_model(sess, model, apply_gradient_op, init_op, imgs, labels, para_list)
                    test_img_batch = []
                    gen_classify_hist += [out_dict]
                    pred_matrix += np.argmax(out_dict['log_y_pred_refine'],1).tolist()
                    rec_err += out_dict['log_rec_err'].tolist()
    # transpose becasue matrix and image different axis notation
    pred_matrix = np.asarray(pred_matrix).reshape(resolution, resolution)
    rec_err = np.asarray(rec_err).reshape(resolution, resolution)
    return pred_matrix, rec_err, gen_classify_hist

if __name__ == '__main__':
    # Step0: configure the experiment
    cfg = {'resolution': 192,
           'scale': [1, .1],
           # 'lenet_ckpt': '/home/exx/Documents/Hope/generative_classifier/assets/pretrained_lenet/experiment_26.ckpt',
           'lenet_ckpt': '/home/exx/Documents/Hope/generative_classifier/original_classifier/models/MNIST/MNIST_2018_02_26_14_08_08/experiment_5.ckpt',
           'vaesub_ckpt': '/home/exx/Documents/Hope/generative_classifier/assets/pretrained_generator/experiment_85.ckpt',
           'gpu_idx': 3,
           }

    FLAGS = tf.app.flags.FLAGS
    tfconfig = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
    )
    tfconfig.gpu_options.allow_growth = True
    cfg['tfconfig'] = tfconfig

    # Step1: Load a pre-trained LeNet
    lenet = Orig_Classifier()
    lenet.build_model(lenet.input)
    trained_model_ckpt = cfg['lenet_ckpt']
    saver = tf.train.Saver()
    sess = tf.Session(config=tfconfig)
    saver.restore(sess, trained_model_ckpt)

    # Step2: load a pre-trained generative model
    with tf.variable_scope('VAE_subnet'):
        model = VAE_subnet()
        model.build_training_model()
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='VAE_subnet'))
        saver.restore(sess, cfg['vaesub_ckpt'])

    # Step3: get attack directions based on LeNet
    orig_img, orig_label, delta_vec, attack_history = get_attack_directions_lenet(lenet, sess)

    pred_res = {}
    with tf.device('gpu:{}'.format(cfg['gpu_idx'])):
        for scale in cfg['scale']:
            cfg['range'] = [-scale, scale, -scale, scale]
            # Step4: get church plot data for lenet
            pred_res['lenet'] = get_church_window_lenet(cfg, lenet, sess, orig_img, delta_vec)
            # Step5: get church plot data for vaesub
            pred_res['vaesub'], rec_err, gen_classify_hist = get_church_window_generator(cfg, model, sess, orig_img, orig_label, delta_vec)
            # Step6: get church plot data for lenet with binary filtering
            pred_res['binary_lenet'] = get_church_window_lenet(cfg, lenet, sess, np.round(orig_img), delta_vec)
            # Step7: church plot
            np.savez('chuch_plot_data_s{}_g{}.png'.format(scale, cfg['gpu_idx']),
                     orig_img = orig_img, deltax = delta_vec['x'], deltay = delta_vec['y'],
                     lenet_pred_matrix = pred_res['lenet'],
                     vaesub_pred_matrix = pred_res['vaesub'],
                     binary_lenet_pred_matrix = pred_res['binary_lenet'],
                     attack_history = attack_history, rec_err = rec_err)
            fn = 'church_plot_s{}_g{}.png'.format(scale, cfg['gpu_idx'])
            church_plt(pred_res, cfg['range'], fn)
            # Step7: critical images

    print('done')