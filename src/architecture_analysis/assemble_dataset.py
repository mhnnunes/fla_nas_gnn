
import pandas as pd
from sys import argv
from os.path import join
from os.path import isdir
from os.path import isfile

modes = ['macro', 'micro']
str_seeds = ['10', '19', '42', '79', '123']
datasets = [
    'cora',
    'citeseer',
    'pubmed',
    'cs',
    'computers',
    'physics',
    'photo'
]
optimizers = ['RL', 'RS', 'ev1003']


def process_arch_string(mode, arch_string):
    return arch_string.split(':')[1].strip() \
        if mode == 'macro' else arch_string.strip()


def arch_start_string(mode):
    # If mode is macro:
    #   Get all lines that start with 'train action:'
    # If mode is micro
    #   Get all lines that start with the string {'action':
    return 'train action:' if mode == 'macro' else "{'action':"


def mode_offset(mode):
    return 1 if mode == 'macro' else 2


def process_log(mode, fullpath):
    file_lines = open(fullpath, 'r').readlines()
    df = pd.DataFrame()
    archs = [(index, process_arch_string(mode, arch_string))
             for index, arch_string in enumerate(file_lines)
             if arch_string.startswith(arch_start_string(mode))]
    # Get val_score from subsequent lines or CUDA error
    val_scores = []
    test_scores = []
    try:
        for index, _ in archs:
            shifted_index = index + mode_offset(mode)
            if shifted_index < len(file_lines):
                val_score = 0.0
                test_score = 0.0
                if 'CUDA out' not in file_lines[shifted_index]:
                    val_score = file_lines[
                        shifted_index].strip().split(',')[0]
                    val_score = float(val_score.split(':')[1])
                    test_score = file_lines[
                        shifted_index].strip().split(',')[1]
                    test_score = float(test_score.split(':')[1])
                val_scores.append(val_score)
                test_scores.append(test_score)
    except Exception as e:
        print(e)
        print(index, val_score)
    assert(len(val_scores) == len(archs))
    assert(len(test_scores) == len(archs))
    df['arch'] = [arch[1] for arch in archs]
    df['val_score'] = val_scores
    df['test_score'] = test_scores
    return df


def process_dataset(source_dir):
    full_df = pd.DataFrame(
        columns=['mode',
                 'seed',
                 'dataset',
                 'optimizer',
                 'arch',
                 'val_score',
                 'test_score'])
    for MODE in modes:
        for SEED in str_seeds:
            for DATASET in datasets:
                for METHOD in optimizers:
                    fullpath = join(source_dir,
                                    MODE + '_results',
                                    'seed_' + SEED,
                                    'results_' + METHOD + '_' + DATASET)
                    print('Processing file: ', fullpath)
                    if isfile(fullpath):
                        file_df = process_log(MODE, fullpath)
                        file_df['mode'] = MODE
                        file_df['seed'] = int(SEED)
                        file_df['dataset'] = DATASET
                        file_df['optimizer'] = METHOD
                        full_df = pd.concat([full_df, file_df],
                                            axis=0,
                                            ignore_index=True)
                    print('DF shape: ', full_df.shape)
    return full_df


if __name__ == "__main__":
    if len(argv) < 3:
        print('ERROR: less than two arguments passed')
    else:
        source_dir = argv[1]
        if not isdir(source_dir):
            print('ERROR: ', source_dir, 'is not a valid directory')
        dataset = process_dataset(source_dir)
        dataset.to_csv(argv[2], index=False, sep=';')
