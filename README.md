# OcuViT

OcuViT is a Hybrid Vision Transformer-Based Approach for Automated Diabetic Retinopathy and Age-Related Macular Degeneration (AMD) Classification. This project utilizes the Vision Transformer (ViT), a state-of-the-art deep learning model, to automatically classify retinal images, helping in the early detection and diagnosis of diabetic retinopathy and AMD.

---

## Files in the Repository

### 1. **DataRead.py**

The `DataRead.py` file handles the process of reading and loading the dataset, which includes retinal images and their corresponding labels for classification. This script ensures that the data is loaded into memory correctly, ready for preprocessing and model training.


### 2. **DataPreprocessing.py**

The `DataPreprocessing.py` file contains the code for preprocessing the image data, which is a crucial step to ensure optimal performance of the model. This script applies transformations like resizing, normalization, and conversion to tensor format for the images.

### 3. **ViT_Model_Setup.py**

The `ViT_Model_Setup.py` file sets up the Vision Transformer (ViT) model. It loads a pretrained ViT model and customizes it for the classification of retinal images into specific classes, such as diabetic retinopathy and AMD.


### 4. **TrainingAndEvaluation.py**

The `TrainingAndEvaluation.py` file defines the process of training and evaluating the ViT model. It includes functions to train the model, evaluate its performance, and save the best-performing model based on test accuracy. The script also tracks several metrics like accuracy, precision, recall, and AUC during both training and testing.

---

## How to Run the Code

### 1. **Install Dependencies**

First, install the required dependencies by running the following command:

```bash
pip install torch torchvision transformers scikit-learn matplotlib seaborn pandas
