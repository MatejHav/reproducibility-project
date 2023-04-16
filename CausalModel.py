import numpy as np
import pandas as pd

class CausalModel:
    def __init__(self, name, model, t_types, requires_epochs):
        self.name = name
        self.model = model
        self.t_types = t_types
        self.requires_epochs = requires_epochs
        self.dosage = 'dosage'
        self.treatment = 'treatment'

    def __str__(self):
        return f"Causal Model {self.name} used for counterfactual dosage extimation."

    def train(self, X, S, T, Y):
        pass

    def predict_dose_response(self, x, t, s):
        pass

    def pe(self, X, Y, treatments, dosages, S, N):
        """
    X: numpy array of shape (N * T * S, F)
    Y: precomputed true values for each sample, shape (N * T * S)
    treatments: List of all the treatments
    dosages: List of all the dosages
    S: interval used to compute all possible dosages
    N: original number of samples
    """
        dosage_block_size = len(
            S)  # every dosage_block_size number of rows in the inputs corresponds to one specific sample + treatment combined with every dosage level
        # find true max Y value for each sample over all its treatment+dosage values
        optimal_t = np.array(
            [np.amax(Y[treatment_block_start:treatment_block_start + self.t_types * dosage_block_size]) for
             treatment_block_start in range(0, X.shape[0], self.t_types * dosage_block_size)])

        # find max predicted Y values for each sample of their optimal treatment+dosage and use that index to find true Y value of that treatment+dosage combo to compare to the true best values above
        optimal_t_hat = np.array([Y[treatment_block_start + np.argmax(
            [self.predict_dose_response(X[index], treatments[index], dosages[index]) for index in
             range(treatment_block_start, treatment_block_start + self.t_types * dosage_block_size)])] for
                                  treatment_block_start in range(0, X.shape[0], self.t_types * dosage_block_size)])

        pe = (optimal_t - optimal_t_hat) ** 2
        return np.sum(pe) / N

    def dpe(self, X, Y, treatments, dosages, S, N):
        """
    X: numpy array of shape (N * T * S, F)
    Y: precomputed true values for each sample, shape (N * T * S)
    treatments: List of all the treatments
    dosages: List of all the dosages
    S: interval used to compute all possible dosages
    N: original number of samples
    """
        dosage_block_size = len(
            S)  # every dosage_block_size number of rows in the inputs corresponds to one specific sample + treatment combined with every dosage level
        # find true max Y value per block
        optimal_s = np.array(
            [np.amax(Y[dosage_block_start:dosage_block_start + dosage_block_size]) for dosage_block_start in
             range(0, X.shape[0], dosage_block_size)])

        # find optimal dosage within each dosage_block, then use the index of that dosage level to index the true outcome value in the Y array of that dosage level
        optimal_s_hat = np.array([Y[dosage_block_start + np.argmax(
            [self.predict_dose_response(X[dosage_block_start], treatments[dosage_block_start], s) for s in S])] for
                                  dosage_block_start in range(0, X.shape[0], dosage_block_size)])
        dpe = (optimal_s - optimal_s_hat) ** 2
        return np.sum(dpe) / N / self.t_types

    def sqrt_mise(self, X, Y, treatments, dosages, S, N):
        """
    X: numpy array of shape (N * T * S, F)
    Y: precomputed true values for each sample, shape (N * T * S)
    treatments: List of all the treatments
    dosages: List of all the dosages
    S: interval used to compute all possible dosages
    N: original number of samples
    """
        delta_S = (S.max() - S.min()) / len(S)
        f = self.predict_dose_response
        Y_hat = np.array([f(row, treatments[index], dosages[index]) for index, row in enumerate(X)])
        mise = delta_S * (Y - Y_hat) ** 2
        return np.sqrt(np.sum(mise) / N / self.t_types)