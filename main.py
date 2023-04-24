import time

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


def run_quick_experiment(data_access, size=100, number_of_epochs=1000, batch_size=128, test_size=3 * 2048):
    dosage_bounds = th.tensor([(0, 1) for _ in range(data_access.T_TYPES)])
    experiment = Experiment([
        CausalDRNet(input_dim=size, hidden_dim=96, output_dim=1, num_layers=3, num_treatments=data_access.T_TYPES,
                    num_strata=3, dosage_bounds=dosage_bounds),
        MLP(input_size=size + 2, num_treatments=data_access.T_TYPES, hidden_layer_units=96),
        TARNET(input_size=size + 1, num_treatments=data_access.T_TYPES, hidden_layer_units=96)
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


def no_hierarchy_experiment(path, size=100, number_of_epochs=1000, batch_size=128, test_size=3 * 2048):
    """
    Removing the treatment layers and instead adding head that are twice as big for every treatment and dosage.
    Hypothesis should be that samples of the same treatment share less layers, making it harder to learn.
    """
    news2_access = NewsDataAccess(data_dir=path, dimension=size, t=2)
    dosage_bounds = th.tensor([(0, 1) for _ in range(news2_access.T_TYPES)])
    experiment = Experiment([
        CausalDRNet(input_dim=size, hidden_dim=96, output_dim=1, num_layers=3, num_treatments=news2_access.T_TYPES,
                    num_strata=3, dosage_bounds=dosage_bounds, hierarchy=False),
        MLP(input_size=size + 2, num_treatments=news2_access.T_TYPES, hidden_layer_units=96),
        TARNET(input_size=size + 1, num_treatments=news2_access.T_TYPES, hidden_layer_units=96)
    ], news2_access)
    return experiment.run(epochs=number_of_epochs, batch_size=batch_size, test_size=test_size)


def run_main_experiments_quick(path, number_of_runs, size=100, number_of_epochs=1000, batch_size=128,
                               test_size=3 * 2048):
    """
    This runs the same experiments but without kNN and CF, as those take the longest.
    """
    news2_access = NewsDataAccess(data_dir=path, dimension=size, t=2)
    news4_access = NewsDataAccess(data_dir=path, dimension=size, t=4)
    news8_access = NewsDataAccess(data_dir=path, dimension=size, t=8)
    news16_access = NewsDataAccess(data_dir=path, dimension=size, t=16)
    custom_access = CustomNewsDataAccess(data_dir=path, dimension=size, t=2)
    number_of_models = 3 # DRNet, TARNet, MLP
    number_of_metrics = 3 # RMISE, DPE, PE
    # Create empty arrays for results - each array has shape of (number_of_models, number_of_metrics)
    result2 = np.zeros((number_of_models, number_of_metrics), dtype='float64')
    result4 = np.zeros((number_of_models, number_of_metrics), dtype='float64')
    result8 = np.zeros((number_of_models, number_of_metrics), dtype='float64')
    result16 = np.zeros((number_of_models, number_of_metrics), dtype='float64')
    custom_result = np.zeros((number_of_models, number_of_metrics))
    for run_number in range(number_of_runs):

        result2 += run_quick_experiment(news2_access, size=size, number_of_epochs=number_of_epochs, batch_size=batch_size,
                                   test_size=test_size).to_numpy()[:, 1:].astype('float64')
        result4 += run_quick_experiment(news4_access, size=size, number_of_epochs=number_of_epochs, batch_size=batch_size,
                                   test_size=test_size).values[:, 1:].astype('float64')
        result8 += run_quick_experiment(news8_access, size=size, number_of_epochs=number_of_epochs, batch_size=batch_size,
                                   test_size=test_size).values[:, 1:].astype('float64')
        result16 += run_quick_experiment(news16_access, size=size, number_of_epochs=number_of_epochs, batch_size=batch_size,
                                     test_size=test_size).values[:, 1:].astype('float64')
        custom_result += run_quick_experiment(custom_access, size=size, number_of_epochs=number_of_epochs,
                                                batch_size=batch_size,
                                                test_size=test_size).values[:, 1:].astype('float64')
        print(f'FINISHED RUN {run_number}')
    result2 /= number_of_runs
    result4 /= number_of_runs
    result8 /= number_of_runs
    result16 /= number_of_runs
    custom_result /= number_of_runs
    models = ['DRNet', 'MLP', 'TARNET']
    result2 = pd.DataFrame(result2, columns=['RMISE', 'DPE', 'PE'])
    result2['Model'] = models
    result4 = pd.DataFrame(result4, columns=['RMISE', 'DPE', 'PE'])
    result4['Model'] = models
    result8 = pd.DataFrame(result8, columns=['RMISE', 'DPE', 'PE'])
    result8['Model'] = models
    result16 = pd.DataFrame(result16, columns=['RMISE', 'DPE', 'PE'])
    result16['Model'] = models
    custom_result = pd.DataFrame(custom_result, columns=['RMISE', 'DPE', 'PE'])
    custom_result['Model'] = models
    return result2, result4, result8, result16, custom_result




if __name__ == '__main__':
    start_time = time.time()
    path = './'
    size = 50
    number_of_epochs = 200
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

    # For a quick check over multiple seeds comment the previous runs and run the following line
    # result2, result4, result8, result16, custom_result = run_main_experiments_quick(path, number_of_runs=5, size=size, number_of_epochs=number_of_epochs, batch_size=batch_size, test_size=test_size)

    hierarchy_result = no_hierarchy_experiment(path, size=size, number_of_epochs=number_of_epochs,
                                               batch_size=batch_size,
                                               test_size=test_size)
    col_names = ['Model', 'News-2', 'News-4', 'News-8', 'News-16', 'Custom News-2']
    results = combine_results([result2, result4, result8, result16, custom_result], col_names=col_names)
    for metric, table in results.items():
        table.to_csv(f'{path}{time.time()}_{metric}.csv')
    hierarchy_result.to_csv(f'{path}{time.time()}_hierarchy.csv')
    print(f'FINISHED. TOOK: {time.time() - start_time} s')
