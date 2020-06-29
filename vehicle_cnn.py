import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_train_data(): 
    path = os.path.join("data", "train.csv")
    return pd.read_csv(path)

def get_test_data():
    path = os.path.join("data", "test.csv")
    return pd.read_csv(path)

def get_all_images():
    path = os.path.join("data", "test.csv")
    

train_data = get_train_data()
test_data = get_test_data()

