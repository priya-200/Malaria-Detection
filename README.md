# Malaria Detection Using Deep Learning

This project uses Convolutional Neural Networks (CNN) to detect malaria-infected cells in microscopic images. The model is trained on a dataset of microscopy images of red blood cells and classifies them as either "parasitized" or "uninfected."


## Project Description

Malaria is a serious and often deadly disease caused by a parasite that infects red blood cells. Early detection of malaria can greatly improve the chances of successful treatment. In this project, we aim to build a deep learning model that can automatically detect malaria from microscopic images of blood smears.

The model is built using TensorFlow and Keras, and it performs image classification by distinguishing between two categories:

- **Parasitized**: Blood cells infected by malaria parasites.
- **Uninfected**: Healthy, non-infected red blood cells.

## Dataset

The dataset used in this project is the **Malaria Cell Images** dataset, which contains 27,558 cell images in two categories: parasitized and uninfected. The images are taken from the [Malaria Dataset by Dr. D. Rajendra](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria).

The dataset is preprocessed to fit into the model, including resizing images, normalization, and splitting into training, validation, and testing sets.

## Installation

To run this project locally, ensure you have the required dependencies installed. You can install them using `pip`:

```bash
pip install tensorflow
pip install matplotlib
pip install numpy
pip install pandas
pip install tensorflow-datasets

```
## Training
The model was trained using the following settings:

1. Optimizer: Adam
2. Loss Function: Binary Cross entropy
3. Metrics: Accuracy
4. Callback : EarilyStopping
5. Batch Size: 32
6. Epochs: 15 (you can adjust this for longer training)

The dataset is split into 80% training, 10% validation, and 10% test sets. The model achieved a validation accuracy of around 51% after 15 epochs.

## Evaluation
You can evaluate the model by running the evaluate_model.py script to get accuracy, loss, and other metrics on the test dataset.

```
python evaluate_model.py
```
## Conclusion
This project successfully demonstrates how deep learning techniques, specifically Convolutional Neural Networks (CNNs), can be applied to medical image analysis for the detection of malaria. The model achieved moderate accuracy, but further improvements can be made by tuning hyperparameters, augmenting the dataset, and exploring more advanced architectures.
