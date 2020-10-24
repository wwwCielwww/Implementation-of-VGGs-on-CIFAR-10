# Implementation of VGG19 and VGG34 Neural Networks on the CIFAR-10 Dataset

Crediting to the Visual Geometry Group, University of Oxford, the series of VGG convolutional neural network was a phenomenon at that time when the paper [*Very Deep Convolutional Networks for Large-Scale Image Recognition*](https://arxiv.org/abs/1409.1556) was published. 

The paper investigated the effect of a CNN's depth on its accuracy in the large-scale image recognition setting. It proposed a novel architecture of neural networks using very small (3x3) convolution filters, which decrease the number of parameters and time needed for computation significantly, pushing the depth to 16-19 weight layers. This resulted in the state-of-the-art results achieved by VGG19 on various classification tasks.

![VGG Structure](https://github.com/wwwCielwww/Implementation-of-VGGs-on-CIFAR-10/blob/main/Model-Structures/vgg.png)

The number 19 in VGG19 stands for the number of layers with trainable weights, i.e.16 convolutional (conv) layers and 3 fully connected (FC) layers. The VGG-19 was originally trained on the ImageNet challenge (ILSVRC) 1000-class classification task. The network takes a (224, 224, 3) RBG image as the input. And for instance, "conv3-256" denotes a 2D convolutional layer with 256 filters of size 3x3, both stride length and padding are 1 (pixel), whereas "maxpool" denotes a max pooling layer with 2x2 filters and 2x2 stride between each set of conv layers. The output of last pooling layer is flattened and in turn, fed into a FC layer with 4096 neurons, whose output is fed into another FC layer with 1000 neurons (remember there are 1000 classes!). All of the layers are activated using the ReLU function. Ultimately, there is a softmax layer which takes advantage of cross entropy loss, which is commonly used in multi-class classification.

In short, the conv layers are the ones that extract the local features, whereas maxpool layers are used to reduce the size of the input images, while FC layers assemble all the features extracted and softmax is used to make the final decision.
