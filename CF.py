from CausalModel import *
from econml.dml import CausalForestDML

class CausalForest(CausalModel):
    def __init__(self, t_types, number_of_trees=100, max_depth=500, min_samples_leaf=500, max_feat=1000):
        super().__init__("CF", CausalForestDML(n_estimators=number_of_trees, max_depth=max_depth,
                                               min_samples_leaf=min_samples_leaf, max_features=max_feat),
                         t_types=t_types, requires_epochs=False)
        self.dosage = 'dosage'

    def __str__(self):
        return f"Causal Model {self.name} used for counterfactual dosage extimation."

    def train(self, X, S, T, Y):
        X[self.dosage] = S
        self.model.fit(Y=Y, X=X, T=T)
        X.drop(columns=[self.dosage], inplace=True)

    def predict_dose_response(self, x, t, s):
        x = pd.Series(x, [f'z{i + 1}' for i in range(len(x))])
        x[self.dosage] = s
        return self.model.effect(X=[x])[0]