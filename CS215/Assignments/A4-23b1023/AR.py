import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg


df = pd.read_csv("parkingLot.csv")

car = df["vehicle_no"]
print("shape of data: ", df.shape)
print("Original Dataset: ", df.head())
print("vehical no", car.shape)