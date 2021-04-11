
import pickle
import argparse
import pandas as pd
from os.path import exists
from ast import literal_eval
from itertools import product
from server_client.evaluator_client import EvaluatorClient


perf_metrics = [
    'prec_macro',
    'prec_micro',
    'prec_weigh',
    'recall_macro',
    'recall_micro',
    'recall_weigh',
    'f_macro',
    'f_micro',
    'f_weigh',
    'avg_epoch_time',
]
test_metrics = [
    'test_loss',
    'test_acc',
] + perf_metrics
train_metrics = [
    'train_acc',
    'val_loss',
    'val_acc',
] + perf_metrics + [
    'train_time'
]
details = [
    'dataset',
    'arch',
    'seed'
]

search_space = 'macro'
n_batches = 1
n_folds = 1


def chunks(lst, n):
    """
    Yield successive n-sized chunks from lst.
    from:
    https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def append_result_to_df(train_results, test_results, results, arch, seed, data):
    cur_train_results = pd.DataFrame(results['train'], index=[0])
    cur_test_results = pd.DataFrame(results['test'], index=[0])
    cur_train_results['dataset'] = data
    cur_test_results['dataset'] = data
    cur_train_results['arch'] = str(arch)
    cur_test_results['arch'] = str(arch)
    cur_train_results['seed'] = seed
    cur_test_results['seed'] = seed
    train_results = pd.concat([train_results, cur_train_results],
                              axis=0,
                              ignore_index=True)
    test_results = pd.concat([test_results, cur_test_results],
                             axis=0,
                             ignore_index=True)
    return train_results, test_results


def parse_args():
    parser = argparse.ArgumentParser(
        description='Start client that evaluates GNN pipelines.')
    parser.add_argument(
        '-b', '--client_batch_size',
        type=int, default=10,
        help='Size of batches of archs that the client should send for eval.')
    parser.add_argument('-s', '--server_url',
                        required=True, type=str,
                        default='http://127.0.0.1:12345',
                        help='URL of the Evaluator Server.')
    parser.add_argument('-o', '--outfile',
                        required=True, type=str,
                        default='output.pkl',
                        help='URL of the Evaluator Server.')
    return parser.parse_args()


def main(client):
    macro_best_archs = pd.read_csv(
        'architecture_analysis/best_by_dataset.csv',
        sep=';',
        header=None,
        names=['dataset', 'arch'])
    # Prepare output files:
    train_out_filename = args.outfile + '_train.csv'
    test_out_filename = args.outfile + '_test.csv'
    pickle_out_filename = args.outfile + '.pkl'
    pipeline_results = []  # Accumulates all results
    # Prepare lists
    seeds = [10, 19, 42, 79, 123]
    arch_seed_pairs = list(
        product(seeds, macro_best_archs.values.tolist()))
    # Batch the evaluation
    for batch in chunks(arch_seed_pairs, args.client_batch_size):
        query_ids = []
        id2arch = {}
        for seed, data_arch in batch:
            dataset, arch_str = data_arch
            arch = literal_eval(arch_str)
            pipeline_id = client.evaluate_pipeline_async(
                client.build_pipeline(
                    actions=arch,
                    search_space=search_space,
                    dataset_name=dataset,
                    random_seed=int(seed),
                    n_folds=n_folds,
                    n_batches=n_batches))
            print("Sent pipeline: ", pipeline_id)
            id2arch[pipeline_id] = (arch, dataset)
            query_ids.append(pipeline_id)
        # Wait for whole batch
        results = client._get_evaluated_multiple(query_ids)
        print('End of batch, received results:', results)
        # Clear results DFs
        train_results = pd.DataFrame(columns=sorted(details + train_metrics))
        test_results = pd.DataFrame(columns=sorted(details + test_metrics))
        for result in results:
            if 'error' in result and result['error'] != {}:
                print('Error evaluating arch: ', id2arch[result['id']])
                print(result['error'])
                continue
            train_results, test_results = \
                append_result_to_df(train_results=train_results,
                                    test_results=test_results,
                                    results=result,
                                    arch=id2arch[result['id']][0],
                                    seed=seed,
                                    data=id2arch[result['id']][1])
            pipeline_results.append(result)
        print('Train results shape: ', train_results.shape)
        print('Test results shape: ', test_results.shape)
        # Checkpoint files:
        print('Checkpointing files to: ', args.outfile, '...')
        if exists(train_out_filename):
            train_results.to_csv(train_out_filename, sep=';',
                                 index=False, header=False, mode='a')
        else:
            train_results.to_csv(train_out_filename, sep=';', index=False)
        if exists(test_out_filename):
            test_results.to_csv(test_out_filename, sep=';',
                                index=False, header=False, mode='a')
        else:
            test_results.to_csv(test_out_filename, sep=';', index=False)
        pickle.dump(pipeline_results,
                    open(pickle_out_filename, 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)
        print('Checkpoint saved!')


if __name__ == '__main__':
    args = parse_args()
    print('Starting connection to server on: ', args.server_url)
    print('Saving output to: ', args.outfile)
    client = EvaluatorClient(args.server_url)
    main(client)
