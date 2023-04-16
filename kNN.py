from CausalModel import *
from sklearn.neighbors import KNeighborsRegressor

class NearestNeighbour(CausalModel):
    def __init__(self, t_types, k=5):
        super().__init__("kNN", None, t_types=t_types, requires_epochs=False)
        self.dosage = 'dosage'
        self.models = {}
        for t in range(t_types):
            self.models[t] = KNeighborsRegressor(n_neighbors=k)

    def __str__(self):
        return f"Causal Model {self.name} used for counterfactual dosage extimation."

    def train(self, X, S, T, Y):
        X[self.dosage] = S
        for t in range(self.t_types):
            self.models[t].fit(y=Y[T == t], X=X.loc[T == t])
        X.drop(columns=[self.dosage], inplace=True)

    def predict_dose_response(self, x, t, s):
        df = pd.DataFrame(columns=[f'z{i + 1}' for i in range(len(x))])
        df.loc[0] = x
        df[self.dosage] = [s]
        return self.models[t].predict(df)[0]