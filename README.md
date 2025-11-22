# Deep Neural Networks from Scratch 
> A hands-on deep learning project featuring a fully custom Python implementation "from scratch" of feedforward depp neural networks (MLPs), along with comparative TensorFlow and PyTorch versions for practice. 
The repository also includes Convolutional Neural Networks (CNNs) and Autoencoders.
They are trained on datasets such as Iris, Wine Quality, MNIST, and a small Cats vs Dogs vision task. 
A Streamlit interface enables interactive experimentation with architectures and hyperparameters, making the project ideal for learning core neural network mechanics and modern DL workflows. 


#### Tables of contents
* [Path tree](#path-tree)
* [Direct links to folders](#direct-links-to-folders) 
* [Installation](#installation)


## Path tree
```
Credit_and_Fraud_Prediction/
├── cat_dog_datasets/
├── computer_vision_with_libraries/
├── pictures/
├── test/
│   ├── iris/
│   └── mnist/
│
├── deepneuralnetworks.py
├── iris.py
├── main.py
├── mnist.py
├── README.md
├── requirements.txt
└── wine_quality.py
```


## Direct links to folders 



## Installation
1. Clone the project:
```
git clone git@github.com:tomcuel/Credit_and_Fraud_Prediction.git
cd Credit_and_Fraud_Prediction
```
2. Create a python virtual environment: 
```
python3 -m venv venv
source venv/bin/activate  # macOS / Linux
```
3. Install the requirements:
```
python3 -m pip -r requirements.txt
```
4. I used those librairies for this project: 
```py
import os
from math import *
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import h5py
from typing import List, Optional, Tuple
from dataclasses import dataclass
from sklearn.datasets import fetch_openml

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, RobustScaler

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from functools import partial
import optuna

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve, f1_score, precision_score, recall_score, , mean_absolute_error, mean_squared_error, log_loss, r2_score, brier_score_loss, , cohen_kappa_score, matthews_corrcoef
from sklearn.calibration import calibration_curve

import streamlit as st
```
5. Make sure to have Jupyter Notebook installed to run the `.ipynb` files


## Overview

classic ML pipeline with data preprocessing, model definition, training, evaluation, and hyperparameter tuning using Optuna if wanted, depending if we're in the test part or not it's done differently 

For the streamlit app, you can choose how you want to use the deep neural network model from scratch, wether with hyperparameter tuning or by hand to then get metrics that you will be able to visualize with graphs and charts within the app for the different datasets available (Iris, Wine Quality, MNIST).





Then, run the Streamlit apps using the following commands :
```
streamlit run paiement_fraud_app.py
streamlit run credit_risk_app.py
```

<table>
  <tr>
    <td style="text-align:center;">
      <img src="./data/pictures/paiement_fraud_app.png" width="400"/>
    </td>
    <td style="text-align:center;">
      <img src="./data/pictures/credit_risk_app.png" width="400"/>
    </td>
  </tr>
</table>

<img src="./Data/Pictures/signup_screen_render.png" alt="signup_screen_render" width="450" height="300"/>
