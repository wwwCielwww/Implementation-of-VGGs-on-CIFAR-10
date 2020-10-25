# Implementation of VGG19 and VGG34 Neural Networks on the CIFAR-10 Dataset

This repository contains my solution to HW2 of the ICRA training, HKU. 

Hope you can enjoy it! ;)

## Introducing VGGs

Crediting to the Visual Geometry Group, University of Oxford, the series of VGG convolutional neural network was a phenomenon at that time when the paper [*Very Deep Convolutional Networks for Large-Scale Image Recognition*](https://arxiv.org/abs/1409.1556) was published. 

The paper investigated the effect of a CNN's depth on its accuracy in the large-scale image recognition setting. It proposed a novel architecture of neural networks using very **small (3x3) convolution filters**, which **decrease the number of parameters and time needed for computation significantly**, pushing the depth to 16-19 weight layers. This resulted in the state-of-the-art results achieved by VGG19 on various classification tasks.

![VGG Structure](https://github.com/wwwCielwww/Implementation-of-VGGs-on-CIFAR-10/blob/main/Model-Structures/vgg.png)

The number 19 in VGG19 stands for the number of layers with trainable weights, i.e.**16 convolutional (conv) layers and 3 fully connected (FC) layers**. The VGG-19 was originally trained on the ImageNet challenge (ILSVRC) 1000-class classification task. The network takes a (224, 224, 3) RBG image as the input. And for instance, "conv3-256" denotes a 2D convolutional layer with 256 filters of size 3x3, both stride length and padding are 1 (pixel), whereas "maxpool" denotes a max pooling layer with 2x2 filters and 2x2 stride between each set of conv layers. The output of last pooling layer is flattened and in turn, fed into a FC layer with 4096 neurons, whose output is fed into another FC layer with 1000 neurons (remember there are 1000 classes!). All of the layers are activated using the ReLU function. Ultimately, there is a softmax layer which takes advantage of cross entropy loss, which is commonly used in multi-class classification.

In short, the conv layers are the ones that extract the local features, whereas maxpool layers are used to reduce the size of the input images, while FC layers assemble all the features extracted and softmax is used to make the final decision.

## Introducing the [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes (airplane, automobile, bird, etc.), with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

The dataset is commonly used to train ML and CV algorithms. Due to the low resolution (32x32), researchers can quickly try different models to see which works.

## Implementation

To install PyTorch, go to https://pytorch.org/get-started/locally/ and follow the instructions.

Since the procedures for building VGG19 and VGG34 network are quite alike, I'll just briefly introduce how to construct a well-functioning program with the former.

------

Firstly, we define the model by creating a VGG class. We initialize it with the designated features and classifier (i.e. FC and Dropout layers), along with the forward function which feeds the input into the network and returns the result for classification. At last, we create an instance of the VGG class using 

```python
net = VGG()
```

To exploit the fast computation speed that parallel computing brings, I use CUDA gpu here. Please note that we'll need to load the parameters into the gpu before the training begins via 

```python
net = net.to(device)
```

------

Then we download the CIFAR-10 dataset through *torchvision.datasets* and load by using the *Dataloader* module in *torch.utils*. For the purpose of data augmentation, we apply methods such as *RandomCrop*, and also normalization to scale the data without distorting the information.

------

Here we use *CrossEntropyLoss* as our loss function and *SGD* (Stochastic Gradient Descent) as the optimizer. Feel free to twist the parameters and play around, or try some different optimizer, e.g. Adam.

------

Finally we may start to train our network! Due to my limited gpu quota, I only use 10 epochs here. Inside each epoch, we zero the loss initially and then iterate over the train dataset. After obtaining the input images and corresponding labels, we zero the parameter gradients, feed the data into the network that we defined before, perform back propagation based on the loss value calculated, and then update the parameters via our optimizer. 

To get a clearer picture of what's going on in our network, we print the average loss every 6250 mini-batches, which can obviously be any other value. 

------

Up until now, we have finished training the network and are ready to perform some evaluations, for instance, by calculating the accuracy on the test dataset.

Since the loss was still decreasing over the last few epochs, I highly suspect that the accuracy would be greater if more epochs were used.

```markdown
[1,  6250] loss: 2.084
[1, 12500] loss: 1.926
[2,  6250] loss: 1.746
[2, 12500] loss: 1.575
[3,  6250] loss: 1.419
[3, 12500] loss: 1.326
[4,  6250] loss: 1.190
[4, 12500] loss: 1.131
[5,  6250] loss: 1.033
[5, 12500] loss: 0.978
[6,  6250] loss: 0.916
[6, 12500] loss: 0.871
[7,  6250] loss: 0.827
[7, 12500] loss: 0.792
[8,  6250] loss: 0.757
[8, 12500] loss: 0.718
[9,  6250] loss: 0.683
[9, 12500] loss: 0.665
[10,  6250] loss: 0.635
[10, 12500] loss: 0.627
Finished Training
```

## Remarks

After implementing both VGG19 and VGG34, we have witnessed the distinct difference between their variation in loss values during the training and also the accuracy measured on the test dataset. 

To me at the beginning, such difference is very confusing since it is generally acknowledged that with a deeper network the accuracy should be much higher. In this case however, VGG34 not only perform much worse than its VGG19 counterpart in terms of the convergence of loss function, but also got 0% accuracy in some of the classes.

Having read the [paper](https://arxiv.org/abs/1512.03385) that my group leader Xiheng Hao has recommended, a degradation problem may account for the circumstance described above: with the network depth increasing, accuracy gets saturated and then degrades rapidly. Unexpectedly, such degradation is not caused by overfitting, and adding more layers to a suitably deep model leads to even higher training error, which is verified by many experiments. Some of the notorious issues might also occurred, such as vanishing/exploding gradients.

Consequently, a new framework (aka. *deep residual learning*) based on the original VGG was introduced in addressing these problems.

