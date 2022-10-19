import numpy as np


class InFair:
    """
    Takes a selection of thresholds and probabilities in order to induce individual fairness from a probabilistic/score
    function that has been optimised for group fairness
    """

    def __init__(self, model, thresholds, probabilities, ignore, constant):
        self.model = model
        self.T = thresholds
        self.p = probabilities
        self.ignore = ignore
        self.constant = constant

        # calculate bounds and interpolants

        self.tau = np.zeros(len(probabilities))
        for i in range(0, len(self.tau)):
            self.tau[i] = self.T[0, i] + (self.T[1, i] - self.T[0, i]) * (1 - self.p[i])

    def predict(self, data, groups):
        """

        :param data: data which we are classifying
        :param group: protected attribute of each input (as integer)
        :return: classification
        """

        # get probabilities from underlying model
        scores = self.model.estimator_.predict_proba(data)[:, 1]
        ans = np.zeros(len(scores))
        for index, (score, group) in enumerate(zip(scores, groups)):
            method = np.random.choice([True, False], p=[self.ignore[group], 1-self.ignore[group]])

            if not method:
                p = self.interp(score, self.T[0, group], self.T[1, group], self.p[group], self.tau[group])

            else:
                p = self.interp(score, 0, 1, self.constant[group], (1-self.constant[group]))

            ans[index] = np.random.choice([1, 0], p=[p, 1 - p])

        return ans

    def interp(self, score, T_0, T_1, p, tau):
        """
        Uses a linear relationship to calculate probabilities between thresholds whilst conserving the average
        probability
        :param T_0: lower threshold
        :param T_1: upper threshold
        :param p: probability
        :param tau: proportion for gradients and bounds
        :param score: score given from underlying predictor
        :param group: which protected group does the input belong to
        :return: probability of belonging to a specific class
        """

        delta_T = T_1 - T_0
        q = 1 - p

        if score < T_0:
            prob = 0
        elif score >= T_1:
            prob = 1
        elif score < tau and score < T_1:
            prob = (p / (delta_T * q)) * (score - T_0)
        else:
            prob = (q / (delta_T * p)) * (score - T_1) + 1

        return prob