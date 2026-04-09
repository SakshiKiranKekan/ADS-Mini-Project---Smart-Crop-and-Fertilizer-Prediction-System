# Importing the libraries needed:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# Reading the dataset for the crops:

crop = pd.read_csv("Crop_recommendation.csv")


# Fetching the first 5 dataframes of the 'crop' dataset:

crop.head()


# Fetching the last 5 dataframes of the 'crop' dataset:

crop.tail()


# Shape and overall information about the crop dataset:

crop.shape
crop.info()


# Analysing the missing values in the dataset:

crop.isnull()


# Total number of missing/null values:

crop.isnull().sum()


# Checking duplicate values:

crop.duplicated().sum()


# Statistics of the dataset:

crop.describe()


# Checking the target feature distribution:

crop['label'].value_counts()


# Feature extraction (excluding label):

features = crop.columns.to_list()
features.remove("label")
features
