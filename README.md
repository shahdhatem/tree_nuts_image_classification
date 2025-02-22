This repository contains a Google Colab notebook (tree_nuts_image_classification.ipynb) that demonstrates the process of classifying images of tree nuts using deep learning techniques. The notebook leverages TensorFlow and Keras to build and train a convolutional neural network (CNN) based on the MobileNetV2 architecture. The dataset used for this project is the "Tree Nuts Image Classification" dataset from Kaggle, which is directly downloaded and used in the notebook.

Introduction
The goal of this project is to classify images of different types of tree nuts using a deep learning model. The notebook walks through the entire process, from loading and preprocessing the dataset to building, training, and evaluating the model. The model is based on the MobileNetV2 architecture, which is a lightweight and efficient CNN that is well-suited for image classification tasks.

Requirements
To run this notebook, you will need the following Python libraries:
TensorFlow
Keras
NumPy
Matplotlib
Kaggle (for downloading the dataset)

These libraries are pre-installed in Google Colab, so no additional installation is required unless specified in the notebook.

Setup
Open the Notebook in Google Colab:
Click the "Open in Colab" button at the top of the notebook file in this repository.
Alternatively, you can upload the notebook to your Google Drive and open it in Colab.

Download the Dataset:
The dataset is downloaded directly from Kaggle using the Kaggle API. You will need to upload your Kaggle API key (kaggle.json) to Colab to access the dataset.
Follow the instructions in the notebook to upload the kaggle.json file and download the dataset.

Run the Notebook:
Execute the cells in the notebook sequentially to load the dataset, preprocess the images, build the model, train it, and evaluate its performance.

Usage
Open the Notebook:
Open the notebook in Google Colab by clicking the "Open in Colab" button or uploading it manually.

Download the Dataset:
Follow the instructions in the notebook to download the dataset using the Kaggle API.

Run the Cells:
Execute the cells in the notebook step-by-step to:
Load and preprocess the dataset.
Build the MobileNetV2-based model.
Train the model using data augmentation.
Evaluate the model on the test set.
Visualize the results.

Dataset
The dataset used in this project is the "Tree Nuts Image Classification" dataset from Kaggle. It contains images of 10 different types of tree nuts, including almonds, cashews, and walnuts. The dataset is already split into training, validation, and test sets.

Training Set: 1,163 images
Validation Set: 50 images
Test Set: 50 images
The dataset is downloaded directly from Kaggle using the Kaggle API, as shown in the notebook.

Model Architecture
The model is based on the MobileNetV2 architecture, which is pre-trained on the ImageNet dataset. The final layers are customized for the tree nuts classification task. The architecture includes:
MobileNetV2 base model (pre-trained on ImageNet)
Global Average Pooling layer
Dense layers with ReLU activation
Softmax output layer for classification

The model is compiled using the Adam optimizer with a learning rate of 0.0001 and categorical cross-entropy loss.
Training
The model is trained for 12 epochs using the training and validation datasets. The training process includes data augmentation techniques such as:Rotation
Width shift
Height shift
Shear
Zoom
Horizontal flip
These techniques help improve the model's generalization ability and prevent overfitting.

Evaluation
After training, the model's performance is evaluated on the test set. The notebook includes code to:
Calculate the test accuracy and loss.
Visualize the model's predictions on a set of randomly selected test images.

Results
The notebook provides detailed results, including:
Training and validation accuracy/loss curves
Test accuracy and loss
Visualizations of the model's predictions on test images

The model achieves a high test accuracy, demonstrating its effectiveness in classifying tree nut imag
