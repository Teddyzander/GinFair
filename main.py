import numpy as np
import data_util.data_util as data_util
from sklearn.linear_model import LogisticRegression

size = 1000
data, target, sens = data_util.gen_data(1000)

# save sensitive dat and then remove it from the data set
sensitive = data[sens]
data = data.drop(sens, axis=1)

print('stop')
