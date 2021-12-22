I created this project in order to learn and understand convolutional neural network for image classification. Its functionality is to decide whether there is a cat or a dog in
the given picture. A simple interface allows you to select a picture for classification.

<p align="center">
  <img src="https://github.com/DJcarlo23/deep-learning-mini-projects/blob/main/Cats%20and%20dogs%20classifier/images/window.PNG?raw=true" alt="Sublime's custom image"/>
</p>

To train the network I used pretrained VGG16 convolutional network which returns vector of 8 192 values. Then these values are used to train another network which is multilayer perceptron. Finally thanks to sigmoid activation function we get probability that there is a dog or cat in the picture.     

