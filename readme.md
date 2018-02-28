# Generative classifier
Making use of generative models to defend adversarial attacks. 
Here Variational Auto-Encoder with multiple decoders is used to test the idea.
Network structure:

![net_struct](../master/assets/net_struct.png)

## Prerequisite
Python2.7, Tensorflow 1.1.0

## Usage

### code structure
* assets
* adversarial_attacks
  > get 
* generative_classifier
    * VAE_subnet.py
      > training of generative model
    * generator_models.py
      > networks structure
* original_classifier
* utils



### pre-train networks
You can find pretrained models in 
/assets/pretrained_generator and /assets/pretrained_lenet

You can also train the discriminator from scratch, by running:

`python ./original_classifier/lenet_mnist.py` 

And to train the generator, by running:

`python ./generative_classifier/VAE_subnet.py` 

To monitor the training process, run:

```
$ tensorboard --logdir=./saved_logs
```

### perform classification

To obtain classification accuracy on MNIST/ adversarial images, run:

`python ./generative_classifier/classification.py` 

To obtain church window plot, run:

`python ./adversarial_attacks/plot_church_window.py` 


## Results
Here is a comparison of the overall accuracy with [binary filtered classifier](https://arxiv.org/abs/1704.01155):

![overall_acc](../master/assets/imgs/overall_acc.png)

Visualization of different defense method of gradient based attack using 
[church window plot](https://arxiv.org/abs/1611.02770):

![church_plt_zoom_out](../master/assets/imgs/church_plot_s4_g1.png)

X axis is the adversarial gradient direction, Y axis is an arbitary orthogonal direction.
Zoom in around origin:

![church_plt_zoom_in](../master/assets/imgs/church_plot_s0.8_g1.png)

 
At the boundary on X axis, critical FGSM images:

![critical_img_lenet](../master/assets/imgs/critical_img_lenet.png)

![critical_img_vaesub](../master/assets/imgs/critical_img_vaesub.png)

## ToDO

-[ ] weight sharing in decoders

-[ ] smallNORB dataset

-[ ] overlapping mnist 