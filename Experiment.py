import pandas as pd


class Experiment:
    def __init__(self, models, data_access):
        self.models = models
        self.data = data_access

    def train(self, epochs, batch_size):
        print('Gathering data...')
        X, S, T, Y = self.data.get_batch(epochs * batch_size)
        print('Training: ')
        no_epoch_models = filter(lambda m: not m.requires_epochs, self.models)
        epoch_models = filter(lambda m: m.requires_epochs, self.models)
        for model in no_epoch_models:
            print(f'\t- {model.name}')
            model.train(X, S, T, Y)
        for model in epoch_models:
            print(f'\t- {model.name}')
            for epoch in range(epochs):
                batch_X = X.sample(batch_size)
                batch_S = S[batch_X.index]
                batch_T = T[batch_X.index]
                batch_Y = Y[batch_X.index]
                model.train(batch_X, batch_S, batch_T, batch_Y)

    def test(self, test_size):
        print('Gathering testing data...')
        X, _, _, _ = self.data.get_batch(test_size)
        # result = pd.DataFrame(columns=['Model', 'RMISE'])
        result = pd.DataFrame(columns=['Model', 'RMISE', 'DPE', 'PE'])
        print('Reformating testing data...')
        new_X, Y, T, S, s, n = self.data.reformat_data(X, self.data.get_true_outcome)
        print('Testing:')
        for model in self.models:
            print(f'\t- {model.name}')
            # result.loc[len(result)] = [model.name, model.sqrt_mise(new_X, Y, T, S, s, n)]
            result.loc[len(result)] = [model.name, model.sqrt_mise(new_X, Y, T, S, s, n),
                                       model.dpe(new_X, Y, T, S, s, n), model.pe(new_X, Y, T, S, s, n)]
        return result

    def run(self, epochs=10, batch_size=1000, test_size=1000):
        self.train(epochs=epochs, batch_size=batch_size)
        print('Finished.')
        return self.test(test_size=test_size)

    def plot(self):
        pass

    def _plot_model(self, model):
        pass
