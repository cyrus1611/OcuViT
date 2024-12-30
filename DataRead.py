
# Import necessary libraries
from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
import keras
from keras.models import Model
from keras.layers import Dense, Input, Flatten, LayerNormalization, Dropout, MultiHeadAttention, Embedding
from keras.layers import Add, GlobalAveragePooling1D
from keras import backend as K
import tensorflow as tf
from keras.layers import Layer
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from transformers import ViTForImageClassification


# File paths
file = "C://Users/acer/Desktop/Widows_Files/Research/APTOS/train_images/"
file1 = "C://Users/acer/Desktop/Widows_Files/Research/APTOS/val_images/"
file2 = "C://Users/acer/Desktop/Widows_Files/Research/APTOS/test_images/"

# Load and preprocess images
def load_images(file_path, target_size=(224, 224)):
    img_files = glob.glob1(file_path, "*.PNG")
    data = []
    for img in img_files:
        image_path = file_path + img.split('/')[-1]
        image = Image.open(image_path).convert('RGB')
        resized_image = image.resize(target_size)
        data.append(np.array(resized_image))
    return np.array(data)

d = load_images(file)
d1 = load_images(file1)
d2 = load_images(file2)

print("Training images shape:", d.shape)
print("Validation images shape:", d1.shape)
print("Test images shape:", d2.shape)

# Concatenate image data
df = np.concatenate((d, d1, d2), axis=0)
print("Total concatenated data shape:", df.shape)
# Normalize pixel values
df = df / 255.0

# Add labels
trb = pd.read_csv("train_1.csv")
vb = pd.read_csv("valid.csv")
tb = pd.read_csv("test.csv")

# Combine labels
labels = np.concatenate((trb["diagnosis"], vb["diagnosis"], tb["diagnosis"]), axis=0)
y = np.array(labels)
assert len(y) == df.shape[0], "Mismatch in labels and data samples!"
print("Labels shape:", y.shape)

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.15, random_state=42)