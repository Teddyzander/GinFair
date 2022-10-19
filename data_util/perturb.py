import numpy as np
import data_util.data_util as data_util
import warnings
from sklearn.linear_model import LogisticRegression
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from fairlearn.postprocessing import ThresholdOptimizer

# go away, warnings
warnings.simplefilter("ignore")

# create the training data
size = 100000
tr_data, tr_target, tr_sensitive = data_util.gen_data(size)

# fit a logistic model
model = LogisticRegression(max_iter=10000)
model.fit(tr_data, tr_target)

# create the testing data
size = 100000
te_data, te_target, te_sensitive = data_util.gen_data(size)

# check fairness and accuracy
output_prob = model.predict_proba(te_data)
output = model.predict(te_data)
acc_score = 1 - np.sum(np.abs(te_target - output)) / size
demo_score = demographic_parity_difference(te_target, output,
                                           sensitive_features=te_sensitive)

print('Accuracy (Base): {}'.format(acc_score))
print('Fairness (Base): {}'.format(demo_score))

# use threshold optimiser to create a fairer model
# fit threshold optimiser to baseline model
opt_model = ThresholdOptimizer(
    estimator=model,
    constraints='demographic_parity',
    objective='accuracy_score',
    prefit=False,
    grid_size=100000)
opt_model.fit(tr_data, tr_target, sensitive_features=tr_sensitive)

fair_output = opt_model.predict(te_data, sensitive_features=te_sensitive)
fair_acc_score = 1 - np.sum(np.abs(te_target - fair_output)) / size
fair_demo_score = demographic_parity_difference(te_target, fair_output,
                                                sensitive_features=te_sensitive)

print('Accuracy (Threshold Optimised): {}'.format(fair_acc_score))
print('Fairness (Threshold Optimised): {}'.format(fair_demo_score))

# create multiple datasets of 'similar individuals'
iters = 1000
prob = 1/3
similar = np.zeros((iters, te_data.shape[0], te_data.shape[1]))
indiv_fair_base = np.zeros(iters)
indiv_fair_opt = np.zeros(iters)
for i in range(0, iters):
    perturb = np.random.laplace(loc=0, scale=0.0005*(i/100), size=te_data.shape)
    similar[i] = te_data + perturb
    sim_output = model.predict(similar[i])
    sim_output_opt = opt_model.predict(similar[i], sensitive_features=te_sensitive)
    indiv_fair_base[i] = np.abs(np.sum(output - sim_output)) / len(output)
    indiv_fair_opt[i] = np.abs(np.sum(fair_output - sim_output_opt)) / len(fair_output)

i_fair_base = np.mean(indiv_fair_base)
i_fair_opt = np.mean(indiv_fair_opt)
print('Individual Fairness (Base): {}'.format(i_fair_base))
print('Individual Fairness (Threshold Optimised): {}'.format(i_fair_opt))
