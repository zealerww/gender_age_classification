# Gnder and Age Classification

Using conv network to classify gender and age.

In paper [Age and Gender Classification using Convolutional Neural Networks](https://www.openu.ac.il/home/hassner/projects/cnn_agegender/CNN_AgeGenderEstimation.pdf), author come up a model of conv netowrk to estimate gender and age.

I reproduce their work in tensorflow and make some improvment.

Dataset: [Adience Benchmark](https://www.openu.ac.il/home/hassner/Adience/data.html#agegender). About 20k images, 2 categories in gender, 8 categories in age.

## Experiments

1. reproduce their work(called model_origin later)
2. using other networks, such as inception-like, resnet-like
3. using bathnormalization, change activation function
4. using pre-trained VGG face network and fine tune fully connected layers

model_origin get 85.9% accuracy in gender and 49.5% in age.

Replacing the model_origin with inception-like/resnet-like netowork don't get significant improvement. Using selu as the activation function can get 2% improvement in gender estimation.

VGG face network is a network for face recogniton, I use it as a face feature extractor. It is a good base model. Using pre-trained conv layers of VGG face network and fine-tuning fully connected layers gets 91% accuracy in gender and 55% in age.

You get get pre-trained VGG face network weight [here](https://pan.baidu.com/s/1F3d1pXROnvTjebSI4fxUmw)

## Others

Multi-cropping can get a little improvement but consuming more time.

## TODOS

- [ ] merge gender and age estimation to one graph
- [ ] upload yolo-face detection code

