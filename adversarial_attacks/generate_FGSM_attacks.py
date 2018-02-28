import tensorflow as tf
from utils.plot_figures import *



def get_attack_directions_lenet(model, sess):
    '''
    get attack direction for a SINGLE image
    :param model:
    :param sess:
    :return:
    '''

    temp = set(tf.all_variables())

    model.input_pl = tf.placeholder(tf.float32, [1, model.imsize, model.imsize, 1], name='input_pl')
    model.input_pl_var = tf.Variable(model.input_pl, name='input_pl_var')
    label_pl = tf.placeholder(tf.float32, [1, model.num_class], name='label_pl')
    output = model.build_model(tf.tile(model.input_pl_var,(model.batch_size,1,1,1)), reuse=True)

    # find the gradient push away from true label
    negative_pred_err = tf.reduce_mean(tf.abs(output['Predictions'] * label_pl))
    optimizer = tf.train.AdamOptimizer(0.01)
    grads = optimizer.compute_gradients(negative_pred_err, var_list=model.input_pl_var)
    apply_gradient_op = optimizer.apply_gradients(grads)
    gradient_direction = tf.gradients(negative_pred_err, model.input_pl_var)[0]
    init_op = tf.initialize_variables(set(tf.all_variables()) - temp)

    orth_transform = np.asarray([[(-1) ** j] * model.imsize for j in range(model.imsize)])

    for i in range(100):
        batch_xs, batch_ys = model.mnist.test.next_batch(1)
        batch_xs = batch_xs.reshape(model.imsize,model.imsize)
        feed_dict = {model.input_pl: batch_xs.reshape(1, model.imsize, model.imsize, 1), label_pl:batch_ys}
        sess.run(init_op, feed_dict)

        delta_vec = {}
        gradient_direction_var = np.squeeze(sess.run(gradient_direction, feed_dict))
        c_grad = np.sum(batch_xs) / np.sum(np.abs(gradient_direction_var))
        delta_vec['x'] = c_grad * gradient_direction_var
        delta_vec['y'] = delta_vec['x'][::-1, ::-1] * orth_transform

        attack_history = {}
        attack_history['x_hist'] = []
        attack_history['y_hist'] = []
        for i in range(50):
            _, adversarial_img, adversarial_pred = sess.run([apply_gradient_op, model.input_pl_var, output['Predictions']], feed_dict)
            attack_history['x_hist'] += [adversarial_img]
            attack_history['y_hist'] += [adversarial_pred]

        pred = {}
        pred['ground_truth'] = batch_ys[0]
        feed_dict_orig = {model.input_pl_var: batch_xs.reshape(1, model.imsize, model.imsize, 1)}
        pred['original'] = sess.run(output['Predictions'], feed_dict_orig)[0]
        feed_dict_attack = {model.input_pl_var: attack_history['x_hist'][-1].reshape(1, model.imsize, model.imsize, 1)}
        pred['attacked'] = sess.run(output['Predictions'], feed_dict_attack)[0]
        feed_dict_step = {model.input_pl_var: (delta_vec['x'] + batch_xs).reshape(1, model.imsize, model.imsize, 1)}
        pred['one_large_step'] = sess.run(output['Predictions'], feed_dict_step)[0]

        # output only when the attack is successful,
        if np.argmax(pred['attacked']) != np.argmax(pred['original']):
            # output only directions
            return batch_xs, batch_ys, delta_vec, attack_history

# sess.run(lenet.end_points['Predictions'], {lenet.input: np.tile(orig_img.reshape(1, 28, 28, 1),(lenet.batch_size,1,1,1))})[0]
