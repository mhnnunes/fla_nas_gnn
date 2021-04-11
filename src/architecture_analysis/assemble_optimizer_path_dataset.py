
import pandas as pd
from sys import argv
from os.path import join
from os.path import isdir
from os.path import isfile

modes = [
    'macro',
    # 'micro'
]
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

unique_macro_archs = pd.read_csv(
    'unique_archs_without_last_dim.txt',
    header=None,
    sep=';').reset_index().rename(columns={0: 'arch'})


def process_arch_string(mode, arch_string):
    if mode == 'macro':
        # Remove the last component of the arch (number of classes)
        # Join into a comma separated string
        x = ','.join(
            arch_string.split(':')[1].strip().split(',')[:-1])
        # Replace special characters
        return x.replace("'", '').replace('[', '').replace(' ', '')
    else:
        return arch_string.strip()


def arch_start_string(mode):
    # If mode is macro:
    #   Get all lines that start with 'train action:'
    # If mode is micro
    #   Get all lines that start with the string {'action':
    return 'train action:' if mode == 'macro' else "{'action':"


def mode_offset(mode):
    # how many lines after the arch descriptor is the arch's fitness
    return 1 if mode == 'macro' else 2


def process_log(mode, fullpath):
    file_lines = open(fullpath, 'r').readlines()
    df = pd.DataFrame()
    # Get all lines in the dataset that contain an architecture
    # Process the architectures and return it in the form expected
    # (comma separated actions, without last element - number of classes)
    archs = [process_arch_string(mode, arch_string)
             for arch_string in file_lines
             if arch_string.startswith(arch_start_string(mode))]
    print('arch strings length: ', len(archs))
    print('arch strings 0: ', archs[0])
    df['arch'] = archs
    df = pd.merge(df, unique_macro_archs, on=['arch'])
    print('df after merging, shape: ', df.shape)
    # Get indexes of architectures
    # df['path'] = \
    #     unique_macro_archs.loc[
    #         unique_macro_archs[0].isin(archs)]['index'].values
    if df.shape[0] != len(archs):
        print(set(archs) - set(
            unique_macro_archs[unique_macro_archs['index'].isin(df['path'])]))
    return df.drop(columns=['arch'])


def process_dataset(source_dir):
    full_df = pd.DataFrame(
        columns=['mode',
                 'seed',
                 'dataset',
                 'optimizer',
                 'index'])
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
