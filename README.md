# Keras_deepLearning_CV

This Repository contains Keras implementation of Some Neural Network architectures organized as folder. I created this repo to train myself deep learning using Keras framework. Thus, part of the codes may be from other peoples implementation or from educational websites. As much as I can I will cite in each file my sources and references.   
The Keras Version used is 2.3.1
OS Ubuntu 20.04.1

### Dataset
The dataset used in this practice is intel-image-classification from kaggle at this [link](https://www.kaggle.com/puneet6060/intel-image-classification).
You can find the details about the details in the kaggle website. But give you a short note the size  of the images is (150, 150, 3).
To give you visual sample the images looks like as follow:
![classification images](sampleImageswiththierclass.png)
The network architecture used for this practice is ResNet50
![ResNet50 architecture](ResNet-50-architecture-26-shown-with-the-residual-units-the-size-of-the-filters-and.png)

### Short description of main components
- identity.py have function to implement identityBlock of ResNet
- convolutionalBlock.py have function to implement identityBlock of ResNet
- resNet.py have all the package neccesary, the first the function which build ResNet50 by concatinating the convolutionalBlock and identityBlock. Then the next step is to load the data and load it using ImageDataGenerator then you are good to go for train and evaluate the model.
