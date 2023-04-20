from CausalModel import *
import torch as th
from torch import nn
from collections import OrderedDict

class TarnetModel(nn.Module):
    """
    TARNET model with output heads for each treatment type.
    """

    def __init__(self, input_size, num_treatments, hidden_layer_units=100, num_layers=3):
        """
      input_size is number of features (without treatment and dosage!) + 1 for additional dosage parameter input
      """
        super(TarnetModel, self).__init__()
        self.num_treatments = num_treatments

        self.base_layers = self.create_base(input_size, hidden_layer_units, num_layers)
        self.treatment_heads = self.create_heads(num_treatments, hidden_layer_units, num_layers)

    def forward(self, x, t):
        """
      Assumes:
      1) x has extra parameter s for dosage inside it
      2) t (treatment type) is list of numbers from 0 to n (for indexing correct output head)
      """
        result = th.zeros(len(t))
        res = self.base_layers(x)
        for treatment in np.random.permutation(self.num_treatments):
            temp_res = self.treatment_heads[int(treatment)](res[t == treatment])
            result[t == treatment] = temp_res[:, 0]
        return result

    def create_base(self, input_size, hidden_layer_units, num_layers):
        modules = []
        for i in range(num_layers):
            modules.append(
                (f"{i}-base-linear", nn.Linear(input_size if i == 0 else hidden_layer_units, hidden_layer_units)))
            modules.append((f"{i}-base-elu", nn.ELU()))
        return nn.Sequential(OrderedDict(modules))

    def create_heads(self, num_treatments, hidden_layer_units, num_layers):
        treatment_blocks = []
        for _ in range(num_treatments):
            modules = []
            for i in range(num_layers):
                modules.append((f"{i}-treatment-linear",
                                nn.Linear(hidden_layer_units, 1 if (i + 1) == num_layers else hidden_layer_units)))
                if (i + 1) != num_layers:
                    modules.append((f"{i}-treatment-elu", nn.ELU()))
            treatment_blocks.append(nn.Sequential(OrderedDict(modules)))
        return nn.ModuleList(treatment_blocks)


class TARNET(CausalModel):
    # In original paper they optimise hyperparameters: batch size, num_units hidden layers, number of hidden layers.
    # Input size is features + 1 for dosage as additional input
    def __init__(self, input_size, num_treatments, hidden_layer_units=100, learn_rate=0.001):
        super().__init__("TARNET", TarnetModel(input_size, num_treatments, hidden_layer_units), t_types=num_treatments,
                         requires_epochs=True)
        self.optimizer = th.optim.Adam(list(self.model.parameters()), lr=learn_rate)
        self.criterion = nn.MSELoss()
        self.dosage = 'dosage'

    def __str__(self):
        return f"TARNET model for causal effect estimation with multiple treatments and dosage levels."

    def train(self, X, S, T, Y):
        X = X.copy()
        X[self.dosage] = S
        X = th.Tensor(X.values)
        Y = th.Tensor(Y)

        self.optimizer.zero_grad()
        outputs = self.model(X, T)
        loss = self.criterion(outputs, Y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict_dose_response(self, x, t, s):
        x = np.append(x.copy(), [s])
        x = th.Tensor(x).unsqueeze(0)
        return self.model(x, [t]).item()
