# Deep Neural Networks from Scratch 
> 


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


## 