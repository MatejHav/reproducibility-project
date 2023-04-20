# General Python libraries
import os
import time
import subprocess
import pandas as pd
import numpy as np

# Neural networks
import torch as th
import torch.nn as nn

# Data processing
import sqlite3
from scipy.special import softmax
from util import *

# kNN
from sklearn.neighbors import KNeighborsRegressor

# PCA
from sklearn.decomposition import PCA


class NewsDataAccess:
    def __init__(self, data_dir, t=2, dimension=100, k=8, run_pca=True):
        self.T_TYPES = t
        self.C = 50
        self.k = k
        self.data_dir = data_dir
        self.db = sqlite3.connect(data_dir + 'news.db')
        self.pca = None
        if run_pca:
            self.pca = PCA(n_components=dimension)
        # Generate a different mu for every cluster for outcome | [(mu, sigma) for each treatment]
        self.clusters = [(N(0.45, 0.15), max(0, N(0.1, 0.05))) for _ in range(t)]
        # For every treatment, generate 2 centroids which are random samples from the data
        self.centroids = []
        print(f"___________NEWS-{t}_______________")
        self.setup(run_pca=run_pca)

    def setup(self, size=8000, run_pca=True):
        print("(1) Setup starting...")
        data = pd.read_sql_query(f"SELECT x, z FROM news ORDER BY RANDOM() LIMIT {size}", self.db)
        data['z'] = data['z'].apply(list)
        n = len(data['z'][0])
        base_data = np.array(data['z'].to_list())
        base_data = pd.DataFrame(base_data, columns=[f"z{i + 1}" for i in range(n)])
        self.original_data = None
        if run_pca:
            print("(2) Running PCA...")
            # First perform PCA
            self.pca.fit(base_data)
            print(f"Variance retained: {np.sum(self.pca.explained_variance_ratio_)}")
            self.original_data = base_data.to_numpy()
            base_data = self.pca.transform(base_data)
            print("(3) PCA finished.")
        # Create 2 centroids for each treatment cluster representing the 0 and 1 dosage
        for t in range(self.T_TYPES):
            samples = base_data[np.random.randint(size, size=2), :]
            self.centroids.append({
                "centroids": [samples[0], samples[1]],
                "mu": N(0.45, 0.15),
                "sigma": N(0.1, 0.05)
            })
        self.base_data = base_data
        self.Y_t = self.generate_potential_treatment_outcome(base_data)
        self.Y_s = self.generate_potential_dosage_outcome(base_data)
        self.T = self.generate_treatment(base_data, self.Y_t, self.Y_s)
        self.S = self.generate_dosage(base_data, self.T, self.Y_t, self.Y_s)
        self.Y = self.generate_outcome(base_data, self.T, self.S, self.Y_t, self.Y_s)
        print(f"({'4' if run_pca else '2'}) Setup finished.")

    def get_batch(self, batch_size=1000):
        """
      Currently I have no idea whether Z is the hidden confounder or treatment.
      So I ignore Z.
      """
        print("*******************************")
        print("(1) Data collection starting...")
        data = pd.read_sql_query(f"SELECT x, z FROM news ORDER BY RANDOM() LIMIT {batch_size}", self.db)
        data['z'] = data['z'].apply(list)
        n = len(data['z'][0])
        batch = np.array(data['z'].to_list())
        batch = pd.DataFrame(batch, columns=[f"z{i + 1}" for i in range(n)])
        print("(2) Data collection finished, preprocessing starting...")
        if self.pca is not None:
            batch = self.pca.transform(batch)
        # Generate the outcome
        print("(3) Outcome generation starting...")
        Y_t = self.generate_potential_treatment_outcome(batch)
        Y_s = self.generate_potential_dosage_outcome(batch)
        T = self.generate_treatment(batch, Y_t, Y_s)
        S = self.generate_dosage(batch, T, Y_t, Y_s)
        Y = self.generate_outcome(batch, T, S, Y_t, Y_s)
        print("(4) Sampling finished.")
        batch = pd.DataFrame(batch, columns=[f"z{i + 1}" for i in range(batch.shape[1])])
        return batch, S, T, Y

    def generate_potential_treatment_outcome(self, X):
        f = lambda sample, treatment: N(self.clusters[int(treatment)][0], self.clusters[int(treatment)][1]) + N(0, 0.15)
        return np.fromfunction(np.vectorize(f), (X.shape[0], self.T_TYPES))

    def generate_potential_dosage_outcome(self, X):
        # Helper function to compute y_s
        def compute_y_s(sample, treatment):
            sample = int(sample)
            treatment = int(treatment)
            distance_to_no_dosage = dist(X[sample], self.centroids[treatment]["centroids"][0])
            distance_to_full_dosage = dist(X[sample], self.centroids[treatment]["centroids"][1])
            soft = softmax([distance_to_no_dosage, distance_to_full_dosage])
            soft_dist0 = soft[0]
            soft_dist1 = soft[1]
            return soft_dist0 * N(self.centroids[treatment]["mu"], self.centroids[treatment]["sigma"]) + soft_dist1 * N(
                self.centroids[treatment]["mu"], self.centroids[treatment]["sigma"])

        f = compute_y_s
        return np.fromfunction(np.vectorize(f), (X.shape[0], self.T_TYPES))

    def generate_treatment(self, X, Y_t, Y_s):
        temp = softmax(self.k * np.multiply(Y_t, Y_s), axis=1)
        return np.apply_along_axis(multinomial, 1, temp)

    def generate_dosage(self, X, T, Y_t, Y_s):
        dosage = np.random.exponential(scale=0.25, size=X.shape[0])
        return dosage

    def generate_outcome(self, X, T, S, Y_t, Y_s):
        f = lambda sample: self.C * Y_t[int(sample)][T[int(sample)]] * Y_s[int(sample)][T[int(sample)]]
        return np.array([f(i) for i in range(X.shape[0])])

    def get_true_outcome(self, x, t, s):
        """
      Provides a the true outcome for any samples point
      Paper:
        1. Find closest neighbour from the original set
        2. Use potential outcomes to recompute the outcome
      """
        index = np.argmin(list(map(lambda y: dist(x, y), self.base_data)))
        return self.C * self.Y_t[index][t] * self.Y_s[index][t]

    def reformat_data(self, X, f, a=0, b=1, precision=16):
        """
      X: all the samples
      f: function converting samples to their real outcomes
      a: start of the S interval
      b: end of the S interval
      precision: number of points between a and b
      """
        N = X.shape[0]
        S = np.linspace(a, b, precision)
        new_X = np.repeat(X.to_numpy(), self.T_TYPES * precision, axis=0)
        # In normal python flatten should not be required, here it somehow is
        dosages = np.repeat([S], N * self.T_TYPES, axis=0).flatten()
        treatments = np.repeat([[i for i in range(self.T_TYPES)]], precision, axis=1)
        treatments = np.repeat([treatments], N, axis=1).flatten()
        Y = f(new_X, treatments, dosages)
        return new_X, Y, treatments, dosages, S, N

    def __str__(self):
        return f'News-{self.T_TYPES}'


class CustomNewsDataAccess(NewsDataAccess):
    def generate_potential_treatment_outcome(self, X):
        return None

    def generate_potential_dosage_outcome(self, X):
        return None

    def generate_treatment(self, X, Y_t, Y_s):
        maximum = X[:, :5].max()
        propensity = X[:, :5].max(axis=1) / (0.01 + maximum)
        return np.vectorize(Bern)(propensity)

    def generate_dosage(self, X, T, Y_t, Y_s):
        x_slice = abs(np.prod(X[:, 5:11], axis=1)) + T
        return nn.Softmax(dim=0)(th.Tensor(x_slice)).numpy()

    def generate_outcome(self, X, T, S, Y_t, Y_s):
        return abs(5 * np.max(X, axis=1) + T * 0.4 / (0.01 + S))

    def get_true_outcome(self, x, t, s):
        index = np.argmin(list(map(lambda y: dist(x, y), self.base_data)))
        return abs(5 * np.max(self.original_data[index]) + t * 0.4 / (0.01 + s))
