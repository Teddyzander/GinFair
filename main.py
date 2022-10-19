import json
import numpy as np
import data_util.data_util as data_util
from InFair.InFair import InFair

from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.interpolate import UnivariateSpline
from scipy import stats
from matplotlib import pyplot as plt

from fairlearn.datasets import fetch_adult, fetch_bank_marketing, fetch_boston
from fairlearn.postprocessing import ThresholdOptimizer, plot_threshold_optimizer
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference, \
    false_positive_rate_difference, true_positive_rate_difference

# run with synthetic data
synth = False

# define fairness constraints
fairness_constraint_metric = demographic_parity_difference
fairness_constraint = 'demographic_parity'

# get data
data = fetch_adult(as_frame=True)
X_raw = data.data
y = (data.target == ">50K") * 1
# y = (data.target == "2") * 1
# y = (data.target.gt(20))

X_raw = X_raw.dropna(axis=0)
only_na = data.data[~data.data.index.isin(X_raw.index)]
y = y.drop(list(only_na.index))

# set sensitive column
# sensitive = 'V1'
# A = (X_raw[sensitive] > 25) * 1
# sensitive = 'B'
# A = X_raw[sensitive].gt(250)
sensitive = 'sex'
A = X_raw[sensitive]

# drop sensitive data
X_raw = X_raw.drop(columns=[sensitive])

if synth:
    X_raw, y, A = data_util.gen_data(100000)

(X_train, X_test, y_train, y_test, A_train, A_test) = train_test_split(
    X_raw, y, A, test_size=0.3, random_state=12345, stratify=y
)

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
A_train = A_train.reset_index(drop=True)
A_test = A_test.reset_index(drop=True)

# set up pipeline
numeric_transformer = Pipeline(
    steps=[
        ("impute", SimpleImputer()),
        ("scaler", StandardScaler()),
    ]
)
categorical_transformer = Pipeline(
    [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, selector(dtype_exclude="category")),
        ("cat", categorical_transformer, selector(dtype_include="category")),
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(max_iter=1000),
        ),
    ]
)

# train threshold optimiser
threshold_optimizer = ThresholdOptimizer(
    estimator=pipeline,
    constraints=fairness_constraint,
    predict_method="predict_proba",
    prefit=False,
)
threshold_optimizer.fit(X_train, y_train, sensitive_features=A_train)
# save values of thresholds and probabilities
thresh_probs = threshold_optimizer.interpolated_thresholder_.interpolation_dict

# get thresholds and probabilities
groups = thresh_probs.keys()
p = np.zeros(len(groups))
T = np.zeros((2, len(groups)))
p_ignore = np.zeros(len(groups))
p_const = np.zeros(len(groups))

for index, key in enumerate(groups):
    T_0 = thresh_probs[key]['operation0'].threshold
    T_1 = thresh_probs[key]['operation1'].threshold
    p[index] = thresh_probs[key]['p0']

    if T_0 > T_1:
        T_0 = T_1
        T_1 = thresh_probs[key]['operation0'].threshold
        p[index] = thresh_probs[key]['p1']

    if T_0 < 0:
        T_0 = 0
    if T_1 > 1:
        T_1 = 1

    T[0, index] = T_0
    T[1, index] = T_1

    if 'p_ignore' in thresh_probs[key]:
        p_ignore[index] = thresh_probs[key]['p_ignore']
        p_const[index] = thresh_probs[key]['prediction_constant']

in_threshold_optimizer = InFair(threshold_optimizer, T, p, p_ignore, p_const)

if not synth:
    bin_A = A_test.map(dict(Female=0, Male=1))
else:
    bin_A = A_test
in_ans = in_threshold_optimizer.predict(X_test, bin_A)

# plot thresholds
# plot_threshold_optimizer(threshold_optimizer)

probs = threshold_optimizer.estimator_.predict_proba(X_test)[:, 1]

sum = 0
labels = np.zeros(len(probs))
for i in range(0, len(probs)):
    if probs[i] > 0.5:
        labels[i] = 1
    sum += abs(labels[i] - y_test[i])

print('Accuracy of f: {}'.format(1 - (sum / len(probs))))

ans = threshold_optimizer.predict(X_test, sensitive_features=A_test)

print('Accuracy of group fair F: {}'.format(1 - (np.sum(np.abs(ans - y_test) / len(ans)))))

print('Accuracy of individually fair F: {}'.format(1 - (np.sum(np.abs(in_ans - y_test) / len(in_ans)))))

print('Group fairness of f: {}'.format(fairness_constraint_metric(y_test, labels, sensitive_features=A_test)))
print('Group fairness of group fair F: {}'.format(fairness_constraint_metric(y_test, ans, sensitive_features=A_test)))
print('Group fairness of individually fair F: {}'.format(
    fairness_constraint_metric(y_test, in_ans, sensitive_features=A_test)))

p, x = np.histogram(probs, bins=100)
x = x[:-1] + (x[1] - x[0]) / 2  # convert bin edges to centers
# plt.plot(x, p / len(probs))
plt.hist(probs, bins=100, density=True, alpha=0.5)
plt.xlim([0, 1])
# plt.show()

sum = 0
iters = 50
for i in range(0, iters):
    X_pert = data_util.randomise(X_test)
    # X_pert = X_test + np.random.choice([0.005, -0.005], p=[0.5, 0.5])
    pert = threshold_optimizer.predict(X_pert, sensitive_features=A_test)
    sum += np.sum(np.abs(pert - labels)) / len(labels)

print('Individual fairness of f: {}'.format(sum / iters))

sum = 0
for i in range(0, iters):
    X_pert = data_util.randomise(X_test)
    # X_pert = X_test + np.random.choice([0.005, -0.005], p=[0.5, 0.5])
    pert = threshold_optimizer.estimator_.predict_proba(X_pert)[:, 1]
    sum += np.sum(np.abs(pert - ans)) / len(ans)

print('Individual fairness of group fair F: {}'.format(sum / iters))

sum = 0
for i in range(0, iters):
    X_pert = data_util.randomise(X_test)
    # X_pert = X_test + np.random.choice([0.005, -0.005], p=[0.5, 0.5])
    pert = in_threshold_optimizer.predict(X_pert, A_test)
    sum += np.sum(np.abs(pert - in_ans)) / len(in_ans)

print('Individual fairness of individually fair F: {}'.format(sum / iters))

print('stop')
