#Libraries
import time
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
import pandas as pd
import seaborn as sns
import math
import sys
import os

start=time.time()

df=pd.read_table('OrionH2.csv', delimiter=',')
sns.heatmap(df)
plt.show()


finish=time.time()
print("Runnig time:",finish-start, "seconds" )
