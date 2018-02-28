import seaborn as sns
from pylab import *
import matplotlib.gridspec as gridspec
import numpy as np


def visualize_generator_performance(classifier, feed_dict, figure_path):
    # Applying encode and decode over test set
    imsize = feed_dict.items()[0][1].shape[1]
    encode_decode = classifier.sess.run(classifier.rec, feed_dict)
    f, a = plt.subplots(3, classifier.examples_to_show, figsize=(10, 2))
    for i in range(classifier.examples_to_show):
        a[0][i].imshow(np.reshape(feed_dict.items()[0][1][i], (imsize, imsize)))
        a[1][i].imshow(np.reshape(encode_decode[i], (imsize, imsize)))
        # a[2][i].imshow(np.reshape(mid_encode_decode[i], (28, 28)))
    f.show()
    f.savefig(figure_path,dpi=imsize*classifier.examples_to_show)


def plot_overall_acc():
    # Imports
    import numpy as np
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6), dpi=80)

    X = np.asarray([15, 25, 40, 55, 75, 100, 125, 150]) / 255.
    C = [1, 0, 0, 0, 16, 16, 17, 21]
    S = [0, 0, 0, 0, 1, 1, 1, 1]
    C = 1 - np.asarray(C) / 272.
    S = 1 - np.asarray(S) / 272.
    C1 = np.asarray([0.95, 0.93, 0.86, 0.82, 0.805, 0.68, 0.62, 0.58])
    S1 = np.asarray([0.96, 0.94, 0.95, 0.91, 0.801, 0.78, 0.72, 0.69])

    # plt.scatter(X, S1, color="red",marker = '*', s=45,label="binarize defense, CS[1,1]")
    # plt.scatter(X, C1, color="red",marker = 'o', s=45, label="binarize defense, CS[0,1]")
    # plt.scatter(X, S, color="blue",marker = '*', s=45, label="generative defense, CS[1,1]")
    # plt.scatter(X, C, color="blue",marker = 'o', s=45, label="generative defense, CS[0,1]")


    # plt.plot(X, (S1 + C1) / 2, color='r', linestyle='dashed', marker='o',
    #          markerfacecolor='r', markersize=10, label="binarize defense, CS")
    # plt.plot(X, (S + C) / 2, color='b', linestyle='dashed', marker='o',
    #          markerfacecolor='b', markersize=10, label="generative defense, CS")

    X = np.asarray([0., 0.1, 0.2, 0.3, 0.4])
    S = np.asarray([ 0.938, 0.909, 0.928, 0.888, 0.852 ])
    S1 = np.asarray([0.9855, 0.946, 0.904, 0.747, 0.531])
    S2 = np.asarray([0.9926, 0.988, 0.971, 0.944, 0.895])

    # plt.plot(X, S2, color='k', linestyle='dashed', marker='v',
    #     markerfacecolor='k', markersize=10, label="reported binarize defense, MNIST")
    plt.plot(X, S1, color='r', linestyle='solid', marker='v',
             markerfacecolor='r', markersize=10, label="binarize defense, MNIST")
    plt.plot(X, S, color='b', linestyle='solid', marker='v',
             markerfacecolor='b', markersize=10, label="generative defense, MNIST")

    # X = np.asarray([15, 150]) / 255.
    # y1 = [0.8367, 0.124]
    # y2 = [1.00, 0.998]
    # plt.scatter(X, y1, color="red", marker='s', s=75, label="binarize defense, TS")
    # plt.scatter(X, y2, color="blue", marker='s', s=75, label="generative defense, TS")

    plt.xlim(-0.01, 0.41)
    plt.ylim(0, C.max() * 1.1)

    plt.xlabel("noise magnitude", fontsize=25)
    plt.ylabel("accuracy", fontsize=25)

    plt.legend(loc='lower left', frameon=False, fontsize=15)
    plt.grid('on')
    plt.show()


def plot_confidence():

    aa=np.load('/home/hope-yao/Downloads/figdata.npz')
    idx = 3#3 6 12 16
    confs = aa['classifications'][idx]
    imgs = aa['features'][idx]


    plt.figure(figsize=(13,8), dpi=80)
    G = gridspec.GridSpec(5, 9)
    ###
    axes_cmp = subplot(G[3:, :])
    x = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    #change#
    y1 = confs[0]
    y2 = confs[1]
    y3 = []
    for i in range(9):
        aa=np.load('/home/hope-yao/Downloads/cmp_mnist/{}.npz'.format(0.05*i))['pred']
        y3 += [aa[9]/np.sum(aa)]


    plt.plot(x, y1, color='k', linestyle='dashed', marker='v',
         markerfacecolor='k', markersize=10, label="no defense")
    plt.plot(x, y2, color='r', linestyle='dashed', marker='o',
         markerfacecolor='r', markersize=10, label="binarize defense")
    plt.plot(x, y3, color='b', linestyle='dashed', marker='*',
         markerfacecolor='b', markersize=10, label="generative defense")
    plt.legend(loc='lower left', frameon=False, fontsize=18)
    plt.xlim(-0.02, 0.42)
    plt.ylim(-0.05, 1.1)
    #x = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    #plt.xticks(x, ('0', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4'), fontsize = 20)
    x = [0, 0.1, 0.2, 0.3, 0.4]
    plt.xticks(x, ('0', '0.1', '0.2', '0.3', '0.4'), fontsize = 20)
    plt.xlabel("FGSM noise level", fontsize=25)
    plt.ylabel("Confidence", fontsize=25)

    ###
    axes_orig = []
    orig = imgs[0]
    for i in range(9):
        axes_orig += [subplot(G[0,i])]
        xticks([]), yticks([])
        plt.imshow(orig[i],interpolation='none')
        if i==0:
            plt.ylabel("Adversarial", fontsize=18)

    ###
    axes_bf = []
    bf = imgs[1]
    for i in range(9):
        axes_bf += [subplot(G[1,i])]
        xticks([]), yticks([])
        plt.imshow(bf[i],interpolation='none')
        if i==0:
            plt.ylabel("binarized", fontsize=18)

    ###
    axes_gen = []
    for i in range(9):
        aa=np.load('/home/hope-yao/Downloads/cmp_mnist/{}.npz'.format(0.05*i))
        gen = aa['img']
        axes_gen += [subplot(G[2,i])]
        xticks([]), yticks([])
        plt.imshow(gen,interpolation='none')
        if i==0:
            plt.ylabel("generation", fontsize=18)

    #plt.savefig('../figures/confidence.png', dpi=64)
    show()

def save_subplots(images, file_path):
    num = len(images)
    plt.figure(figsize=(4,8), dpi=80)
    G = gridspec.GridSpec(1, num)
    axes_gen = []

    for i, img in enumerate(images):
        axes_gen += [subplot(G[0,i])]
        xticks([]), yticks([])
        plt.imshow(img,interpolation='none')
    plt.show()
    plt.savefig(file_path)


def church_plt(pred_res, axis_range, fn=None):
    plt.figure()
    G = gridspec.GridSpec(1, len(pred_res))
    cnt = 0
    for mat_name, pred_matrix in pred_res.items():
        subplot(G[0, cnt])
        cmap = plt.get_cmap('jet', 10)
        cmap.set_under('gray')
        plt.imshow(pred_matrix, cmap=cmap, interpolation='bicubic')
        plt.xticks([0, pred_matrix.shape[0] / 2, pred_matrix.shape[0]-1],
                   ('{}'.format(axis_range[0]), 0, '{}'.format(axis_range[1])), fontsize=20)
        plt.yticks([0, pred_matrix.shape[1] / 2, pred_matrix.shape[1]-1],
                   ('{}'.format(axis_range[2]), 0, '{}'.format(axis_range[3])), fontsize=20)
        plt.colorbar()
        plt.title(mat_name)
        cnt += 1

    if fn:
        plt.savefig(fn, dpi=pred_matrix.shape[0])
    print('done')

if __name__ == '__main__':
    plot_overall_acc()
    print('done')