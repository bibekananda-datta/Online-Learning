# Online-Learning
Repository of MATLAB functions and scripts for online learning algorithms.

## Pre-requisite
Your computer should have some version of MATLAB installed. These codes were developed using MATLAB 2017b, but any recent version will be able to run these codes on any operating system (Windows, Linux, MacOS).

## Data-set
MNIST data found at https://www.kaggle.com/zalando-research/fashionmnist#fashion-mnist_test.csv (**fashion-mnist_train.csv** and **fashion-mnist_test.csv**) was used to test and validate the functions and scripts. Original dataset and detailed description is available at https://github.com/zalandoresearch/fashion-mnist. Please make sure to read details about the dataset and download them from above websites before you try any implemented algorithm or test program.

The training dataset comprises of the pixel information of 60000, 28 x 28 images, each stored as the columns of a matrix of size 60000 x 784. The test dataset comprises of 10000, 28 x 28 images, each stored as the columns of a matrix of size 10000 x 784. For both of the files the first column is the class labels, from 0 to 9.Each training and test example is assigned to one of the following labels:

0: T-shirt/top, 1: Trouser, 2: Pullover, 3: Dress, 4: Coat, 5: Sandal, 6: Shirt, 7: Sneaker, 8: Bag, 9: Ankle boot.

## Algorithms
Perceptron, Passive-Aggressive, Averaged Perceptron are the algorithms used to developed both binary and multiple classifiers. Please read **OnlineAlgorithms.pdf** to see the descriptions of the algorithms. In addition to the algorithms, this .pdf file describes two different problem statements which are solved using the test script files, listed below.

## Description of the .m file

### Function files
To learn more about the function files, please read the comments in the file respective files

**binpercept.m:** Function file based on perceptron algorithm for binary classifiers.

**binpassagg.m:** Function file based on passive-aggressive algorithm for binary classifiers.

**binavgpercept.m:** Function file based on averaged-perceptron algorithm for binary classifiers.

**bintestfun.m:** Function file to evaluate test data for binary classifiers.

**multipercept.m:** Function file based on perceptron algorithm for multiple classifiers.

**multipassagg.m:** Function file based on passive-aggressive algorithm for multiple classifiers.

**multiavgpercept.m:** Function file based on averaged-perceptron algorithm for multiple classifiers.

**multitestfun.m:** Function file to evaluate test data for multiple classifiers.


### Test scripts
These test scripts are designed to demonstrate the capability of function files developed for online learning.

**binaryclasstest.m:** script file to test the functions developed for binary classifier.

**multiclasstest.m** script file to test the functions developed for multiple classifier.
