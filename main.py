
import data_util.data_util as data_util
from sklearn.linear_model import LogisticRegression

# create the training data
size = 100000
tr_data, tr_target, tr_sensitive = data_util.gen_data(size)

# fit a logistic model
model = LogisticRegression(max_iter=10000)
model.fit(tr_data, tr_target)

print('stop')
