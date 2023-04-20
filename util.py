import numpy as np
import pandas as pd


# Normal distribution
def N(mu, sigma):
    return sigma * np.random.randn() + mu


# Bernoulli distribution
def Bern(p):
    return 1 if np.random.rand() <= p else 0


def multinomial(p):
    return np.argmax(np.random.multinomial(1, p))


# Compute euclidean distance
def dist(X, Y):
    return np.linalg.norm(X - Y)

def combine_results(results, col_names):
    metrics = ['RMISE', 'DPE', 'PE']  # Add PE/DPE to metrics list if we have those!

    final_dataframes = {}
    for metric in metrics:
        metric_results = pd.DataFrame(columns=col_names)
        for model in list(results[0]['Model'].unique()):
            vals = [model]
            for res in results:
                vals.append(res.loc[res['Model'] == model][metric].to_numpy()[0])
            metric_results.loc[len(metric_results)] = vals
        final_dataframes[metric] = metric_results

    return final_dataframes
