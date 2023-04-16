from DataAccess import *
from kNN import *
from MLP import *
from TARNet import *
from DRNet import *
from CF import *
from Experiment import *


def run_experiment(data_access, size=100, number_of_epochs=1000, batch_size=128, test_size=3 * 2048):
    dosage_bounds = th.tensor([(0, 1) for _ in range(data_access.T_TYPES)])
    experiment = Experiment([
        CausalDRNet(input_dim=size, hidden_dim=96, output_dim=1, num_layers=3, num_treatments=data_access.T_TYPES,
                    num_strata=3, dosage_bounds=dosage_bounds),
        MLP(input_size=size + 2, num_treatments=data_access.T_TYPES, hidden_layer_units=96),
        TARNET(input_size=size + 1, num_treatments=data_access.T_TYPES, hidden_layer_units=96),
        CausalForest(t_types=data_access.T_TYPES, number_of_trees=8, max_feat=5),
        NearestNeighbour(t_types=data_access.T_TYPES, k=10)
    ], data_access)
    return experiment.run(epochs=number_of_epochs, batch_size=batch_size, test_size=test_size)


def news2_experiment(path, size=100, number_of_epochs=1000, batch_size=128, test_size=3 * 2048):
    news2_access = NewsDataAccess(data_dir=path, dimension=size, t=2)
    return run_experiment(news2_access, size=size, number_of_epochs=number_of_epochs, batch_size=batch_size,
                          test_size=test_size)


def news4_experiment(path, size=100, number_of_epochs=1000, batch_size=128, test_size=3 * 2048):
    news4_access = NewsDataAccess(data_dir=path, dimension=size, t=4)
    return run_experiment(news4_access, size=size, number_of_epochs=number_of_epochs, batch_size=batch_size,
                          test_size=test_size)


def news8_experiment(path, size=100, number_of_epochs=1000, batch_size=128, test_size=3 * 2048):
    news8_access = NewsDataAccess(data_dir=path, dimension=size, t=8)
    return run_experiment(news8_access, size=size, number_of_epochs=number_of_epochs, batch_size=batch_size,
                          test_size=test_size)


def news16_experiment(path, size=100, number_of_epochs=1000, batch_size=128, test_size=3 * 2048):
    news16_access = NewsDataAccess(data_dir=path, dimension=size, t=16)
    return run_experiment(news16_access, size=size, number_of_epochs=number_of_epochs, batch_size=batch_size,
                          test_size=test_size)


def custom_news2_experiment(path, size=100, number_of_epochs=1000, batch_size=128, test_size=3 * 2048):
    custom_access = CustomNewsDataAccess(data_dir=path, dimension=size, t=2)
    return run_experiment(custom_access, size=size, number_of_epochs=number_of_epochs, batch_size=batch_size,
                          test_size=test_size)


if __name__ == '__main__':
    start_time = time.time()
    path = './'
    size = 100
    number_of_epochs = 1000
    batch_size = 128
    test_size = 5000
    result2 = news2_experiment(path, size=size, number_of_epochs=number_of_epochs, batch_size=batch_size,
                               test_size=test_size)
    result4 = news4_experiment(path, size=size, number_of_epochs=number_of_epochs, batch_size=batch_size,
                               test_size=test_size)
    result8 = news8_experiment(path, size=size, number_of_epochs=number_of_epochs, batch_size=batch_size,
                               test_size=test_size)
    result16 = news16_experiment(path, size=size, number_of_epochs=number_of_epochs, batch_size=batch_size,
                                 test_size=test_size)
    custom_result = custom_news2_experiment(path, size=size, number_of_epochs=number_of_epochs, batch_size=batch_size,
                                            test_size=test_size)
    col_names = ['Model', 'News-2', 'News-4', 'News-8', 'News-16', 'Custom News-2']
    results = combine_results([result2, result4, result8, result16, custom_result], col_names=col_names)
    for metric, table in results.items():
        table.to_csv(f'{path}{time.time()}_{metric}.csv')
    print(f'FINISHED. TOOK: {time.time() - start_time} s')
