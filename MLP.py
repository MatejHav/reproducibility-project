from CausalModel import *
import torch as th
from torch import nn
from collections import OrderedDict

class MlpModel(nn.Module):
    """
    MLP with treatment and dosages as additional inputs.
    """

    def __init__(self, input_size, hidden_layer_units=100, num_layers=3):
        """
      input_size is number of features + 1 for dosage + 1 for treatment
      """
        super(MlpModel, self).__init__()
        self.base_layers = self.create_base(input_size, hidden_layer_units, num_layers)

    def forward(self, x):
        """
      Assumes x contains data features + dosage s + treatment t
      """
        return self.base_layers(x)

    def create_base(self, input_size, hidden_layer_units, num_layers):
        modules = []
        for i in range(num_layers):
            modules.append((f"{i}-base-linear", nn.Linear(input_size if i == 0 else hidden_layer_units,
                                                          1 if (i + 1) == num_layers else hidden_layer_units)))
            if (i + 1) != num_layers:
                modules.append((f"{i}-treatment-elu", nn.ReLU()))
        return nn.Sequential(OrderedDict(modules))


class MLP(CausalModel):
    def __init__(self, input_size, num_treatments, hidden_layer_units=100, learn_rate=0.001):
        super().__init__("MLP", MlpModel(input_size, hidden_layer_units), t_types=num_treatments, requires_epochs=True)
        self.optimizer = th.optim.Adam(list(self.model.parameters()), lr=learn_rate)
        self.criterion = nn.MSELoss()
        self.dosage = 'dosage'
        self.treatment = 'treatment'

    def __str__(self):
        return f"MLP model for causal effect estimation with multiple treatments and dosage levels."

    def train(self, X, S, T, Y):
        X = X.copy()
        X[self.dosage] = S
        X[self.treatment] = T
        X = th.Tensor(X.values)
        Y = th.Tensor(Y)

        self.optimizer.zero_grad()
        outputs = self.model(X)[:, 0]
        loss = self.criterion(outputs, Y)
        loss.backward()
        self.optimizer.step()
        return loss

    def predict_dose_response(self, x, t, s):
        x = np.append(x.copy(), [t, s])
        x = th.Tensor(x)
        return self.model(x).item()