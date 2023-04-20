import torch as th
from torch import nn
from CausalModel import *
from collections import OrderedDict

class DRNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_treatments, num_strata, dosage_bounds, hierarchy=True):
        super(DRNet, self).__init__()
        self.num_layers = num_layers
        self.num_treatments = num_treatments
        self.num_strata = num_strata
        self.dosage_bounds = dosage_bounds

        self.base_block = self.get_block(input_dim, hidden_dim, hidden_dim, num_layers)
        self.hierarchy = hierarchy

        if hierarchy:
            self.treatment_block = nn.ModuleList([
                self.get_block(hidden_dim, hidden_dim, hidden_dim, num_layers)
                for _ in range(num_treatments)
            ])

        self.head_block = nn.ModuleList([
            nn.ModuleList([
                self.get_block(hidden_dim, hidden_dim, output_dim, num_layers, use_final_activation=False)
                for _ in range(num_strata)
            ])
            for _ in range(num_treatments)
        ])

    def forward(self, x, t, dos_idx):
        """
        Assume X is a tensor and t and dos_idx are scalars
        """
        # dos_idx = self.get_dosage_idx(s, t)
        x = self.base_block(x)
        if self.hierarchy:
            x = self.treatment_block[t](x)
        x = self.head_block[t][int(dos_idx)](x)
        return x

    # def get_dosage_idx(self, s, t):
    #     a, b = self.dosage_bounds[t]
    #     dosage_idx = math.floor(self.num_strata * ((s - a) / (b - a)))
    #     if s == b:
    #         dosage_idx = self.num_strata - 1
    #     return dosage_idx

    def get_block(self, input_dim, hidden_dim, output_dim, num_layers, use_final_activation=True):
        modules = []
        for i in range(num_layers):
            from_dim = input_dim if i == 0 else hidden_dim
            to_dim = output_dim if i == num_layers - 1 else hidden_dim
            modules.append((f"{i}-linear", nn.Linear(from_dim, to_dim)))
            modules.append((f"{i}-relu", nn.ReLU()))
        if not use_final_activation:
            modules.pop()
        return nn.Sequential(OrderedDict(modules))


class CausalDRNet(CausalModel):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_treatments, num_strata, dosage_bounds,
                 learn_rate=0.0001, hierarchy=True):
        super().__init__("DRNet", DRNet(input_dim, hidden_dim, output_dim, num_layers, num_treatments, num_strata,
                                        dosage_bounds, hierarchy=hierarchy),
                         t_types=num_treatments, requires_epochs=True)
        self.optimizer = th.optim.Adam(list(self.model.parameters()), lr=learn_rate)
        self.criterion = nn.MSELoss()
        self.dosage_bounds = dosage_bounds
        self.num_strata = num_strata
        self.dosage = 'dosage'
        self.treatment = 'treatment'

    def train(self, X, S, T, Y):
        # DRNet can only take batches where the treatment and dosage index are the same
        # (num_treatment, num_strata, batch_size, num_feature)
        X = th.Tensor(X.values)
        S = th.Tensor(S)
        T = th.Tensor(T)
        Y = th.Tensor(Y).unsqueeze(dim=1)
        loss = th.tensor(0)
        for t in np.random.permutation(self.t_types):
            a, b = self.dosage_bounds[t]
            divs = np.linspace(a, b, self.num_strata + 1)
            for dos_idx, (lower, upper) in enumerate(zip(divs, divs[1:])):
                mask = (S >= lower) & (S <= upper) & (T == t)
                if mask.any():
                    X_filtered = X[mask]

                    self.optimizer.zero_grad()
                    outputs = self.model(X_filtered, t, dos_idx)
                    # Adding all the losses
                    loss = loss + self.criterion(outputs, Y[mask])
        loss = loss / (self.t_types * self.num_strata)
        loss.backward()
        self.optimizer.step()
        return loss.item()



    def predict_dose_response(self, x, t, s):
        x = th.Tensor(x)
        return self.model(x, t, s).item()