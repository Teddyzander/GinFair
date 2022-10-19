import numpy as np
import pandas as pd


def gen_data(size):
    """
    Creates the data set
    :param size: number of instances in the data set
    :return: data
    """

    # seed randomness for repeated experiments
    np.random.seed(123)

    # allocate memory for data
    data = np.zeros((size, 4))
    target = np.zeros(size)

    # create data set
    for row in range(0, size):
        # allocate protected group status
        pro_group = np.random.choice([0, 1])
        data[row, 3] = pro_group

        # if we are in the disadvantaged group, we should be disadvantaged
        mult = 1
        if pro_group == 0:
            mult = 0.9

        # create data
        x_1 = np.random.laplace(0.55 * mult, 1 / mult)
        x_2 = np.random.laplace(0.55 * mult, 1 / mult)
        x_3 = np.random.laplace(0.55 * mult, 1 / mult)
        data[row, 0] = x_1
        data[row, 1] = x_2
        data[row, 2] = x_3

        # give target via causality to feature variables (average is probability of being classed as 1)
        prob = (x_1 + x_2 + x_3) / 3
        if prob > 1:
            prob = 1
        elif prob < 0:
            prob = 0

        target[row] = 0
        if prob > 0.5 and pro_group == 1:
            target[row] = np.random.choice([0, 1], p=[0.1, 0.9])
        elif prob > 0.5 and pro_group == 0:
            target[row] = np.random.choice([0, 1], p=[0.3, 0.7])

    # normalise the data
    for i in range(0, data.shape[1]):
        data[:, i] = (data[:, i] - np.min(data[:, i])) / (np.max(data[:, i]) - np.min(data[:, i]))

    # Make into data frame
    data = pd.DataFrame(data,
                        columns=['x_1', 'x_2', 'x_3', 'sens'])

    # make the protected feature discrete
    for col in ['sens']:
        data[col] = data[col].astype(int).astype('category')

    target = pd.Series(target)

    sens = 'sens'

    # save sensitive dat and then remove it from the data set
    sensitive = data[sens]
    data = data.drop(sens, axis=1)

    return data, target, sensitive


def randomise(data):

    data_e = data.copy()
    shape = data.shape
    for i in range(0, shape[0]):
        cat = np.random.choice(np.arange(0, shape[1], dtype=int))
        label = np.random.choice(data.iloc[:, cat])
        data_e.iloc[i, cat] = label

    return data_e
