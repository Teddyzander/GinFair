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
            mult = 0.715

        # create data
        x_1 = np.random.laplace(0.75 * mult, 0.075 / mult)
        x_2 = np.random.laplace(0.75 * mult, 0.075 / mult)
        x_3 = np.random.laplace(0.75 * mult, 0.075 / mult)
        data[row, 0] = x_1
        data[row, 1] = x_2
        data[row, 2] = x_3

        # give target via causality to feature variables (average is probability of being classed as 1)
        prob = (x_1+x_2+x_3) / 3
        if prob > 1:
            prob = 1
        elif prob < 0:
            prob = 0

        target[row] = np.random.choice([0, 1], p=[1 - prob, prob])

    # normalise the data
    for i in range(0, data.shape[1]):
        data[:, i] = (data[:, i] - np.min(data[:, i])) / (np.max(data[:, i]) - np.min(data[:, i]))

    # Make into data frame
    data = pd.DataFrame(data,
                        columns=['x_1', 'x_2', 'x_3', 'sens'])

    # make the protected feature discrete
    for col in ['sens']:
        data[col] = data[col].astype('category')

    sens = 'sens'

    return data, target, sens
