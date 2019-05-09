# Images-classification-overview
Leveraging Machine Learning and Deep Learning to classify images with 20 different classes

## Introduction

In the Machine Learning landscape, image classification has been a problem widely studied. This paper aims at making a **comprehensive review** of the **different approaches** that can be chosen to face this problem. It will involves images understanding, features extraction from images as well as modulation and optimization. We will conclude our study with the implementation of the current state of the art for image classification, convolutional neural network.

The dataset used for the following analysis is composed of 1480 images from **20 different classes**. The classes are not equaly balanced as you can see bellow. The challenge at hand will therefore be to beat the baseline model that always predicts the most frequent class, namely Gorilla.

[img]

## Feature Engineering

In order to approach this challenge from a Machine Learning perspective, it requires an extended Feature Engineering process. We need to determine which information from the image can be useful to discriminate the different classes from one another. During our study, we started with basic feautures such as means and standart deviations for the three colors. Then, we explored more advanced features specific for images analysis, involving contour detection, blurness analysis as well as computation of the color gradient matrice. Here I give the explaination for a couple of them.

### Getting the keypoints of the images

To extract the keypoints out of the images, we are using **KAZE features**. It is a multiscale feature detection algorithm in nonlinear scale spaces. The approach here is to describe 2D features in a non linear scale space by mean of non linear diffusion filtering. This enables us to make bluring locally adaptive to the image data. Once the matrix was built, we computed several statistics on it to get some quantitative new features. As can be seen later on the correlation matrix, they proved to be well explanable of our target feature. 

### Images color gradient matrices

We computed three matrix per axis (one per color). Each matrix represents the color gradient along either the x-axis or the y-axis. Then, we realized a maximum wise operation per axis (using the three color matrix per axis). The idea is to emphasize gradients variation without taking into account the specific color, to focus instead on the shapes within the image.

We then created eight features using the "maximum" gradient matrix on the x axis, by splitting our images in four quarters, and computing the average intensity of the gradient and its sum on all quarters. You can visualize bellow a visualization of one x-axis gradient matrice for an image of plane. 

[img]

### Correlation between the features previously created

[img]

[img]

## Machine Learning models

After carefully selecting the best features for our analysis, we tried some general Machine Learning models to estimate how well can we predict unseen images based on those features. 

## Deep Learning model

We ended our study by looking at some state of the arts approach for image classification problem. Recently, the most used approach has been convolutional neural network or tranfer learning based cnn. As the result, this is the architecture we have chosen for our final model. We have chosen to keep it simple as the goal of this project is not to beat state of the art accuracies but to make an overview of the different approaches for image classification problem. We can visualize bellow the cnn architecture, from [1]. 

[img]

Our simple model is:
- 2D CNN layer
- Max pooling
- 2D CNN layer
- Max pooling
- Dense layer
- Dense layer with 20 neurons (the number of classes)

The model is compiled with categorical cross-entropy loss, using adam optimizer. The model easily beat the different Machine Learning approaches previously tested. 




