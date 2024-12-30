# OcuViT

OcuViT is a Hybrid Vision Transformer-Based Approach for Automated Diabetic Retinopathy and Age-Related Macular Degeneration (AMD) Classification. This project utilizes the Vision Transformer (ViT), a state-of-the-art deep learning model, to automatically classify retinal images, helping in the early detection and diagnosis of diabetic retinopathy and AMD.

---

## Files in the Repository

### 1. **DataRead.py**

The `DataRead.py` file handles the process of reading and loading the dataset, which includes retinal images and their corresponding labels for classification. This script ensures that the data is loaded into memory correctly, ready for preprocessing and model training.

#### Key Features:
- **Dataset Loading**: Loads the training and test datasets.
- **Custom Dataset Class**: Implements a custom PyTorch `Dataset` class for handling images and their labels.
- **DataLoader Instances**: Utilizes PyTorch's `DataLoader` to create batch loaders for both training and testing datasets.

### 2. **DataPreprocessing.py**

The `DataPreprocessing.py` file contains the code for preprocessing the image data, which is a crucial step to ensure optimal performance of the model. This script applies transformations like resizing, normalization, and conversion to tensor format for the images.

#### Key Features:
- **Transformation Pipeline**: Resizes images to 224x224, converts them to tensor format, and normalizes the pixel values using ImageNet statistics.
- **Data Augmentation**: Can easily incorporate augmentation techniques such as rotation, flipping, and color adjustment.
- **Normalization**: Standardizes the data by normalizing pixel values to match the pretrained ViT model’s input requirements.

### 3. **ViT_Model_Setup.py**

The `ViT_Model_Setup.py` file sets up the Vision Transformer (ViT) model. It loads a pretrained ViT model and customizes it for the classification of retinal images into specific classes, such as diabetic retinopathy and AMD.

#### Key Features:
- **Pretrained ViT Model**: Loads the pretrained ViT model from HuggingFace’s `transformers` library.
- **Model Customization**: Modifies the output layer to classify images into 5 classes specific to this task.
- **Device Setup**: Automatically detects whether a GPU is available and moves the model to the appropriate device (CPU or GPU) for efficient training and evaluation.

### 4. **TrainingAndEvaluation.py**

The `TrainingAndEvaluation.py` file defines the process of training and evaluating the ViT model. It includes functions to train the model, evaluate its performance, and save the best-performing model based on test accuracy. The script also tracks several metrics like accuracy, precision, recall, and AUC during both training and testing.

#### Key Features:
- **Training Loop**: A custom `train_epoch` function that trains the ViT model by minimizing the loss.
- **Evaluation Metrics**: Calculates accuracy, precision, recall, and AUC on the test set after each epoch.
- **Early Stopping**: Implements early stopping to prevent overfitting by halting training if the test accuracy does not improve for a certain number of epochs.
- **Model Saving**: Saves the model with the best test accuracy during training.
- **Confusion Matrix**: Generates a confusion matrix to visualize the classification performance.

---

## How to Run the Code

### 1. **Install Dependencies**

First, install the required dependencies by running the following command:

```bash
pip install torch torchvision transformers scikit-learn matplotlib seaborn pandas
