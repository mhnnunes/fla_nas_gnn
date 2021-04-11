
import scipy
import pickle
import argparse
import numpy as np
from umap import UMAP
# import pandas as pd
from tqdm import tqdm
from os.path import exists
# from ast import literal_eval
from itertools import product
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder


macro_search_space = {
    'attention_type': ['gat', 'gcn', 'cos', 'const', 'gat_sym',
                       'linear', 'generalized_linear'],
    'aggregator_type': ['sum', 'mean', 'max', 'mlp'],
    'activate_function': ['sigmoid', 'tanh', 'relu', 'linear',
                          'softplus', 'leaky_relu', 'relu6', 'elu'],
    'number_of_heads': sorted([1, 2, 4, 6, 8, 16]),
    'hidden_units': sorted([4, 8, 16, 32, 64, 128, 256])
}

macro_action_list = ['attention_type',
                     'aggregator_type',
                     'activate_function',
                     'number_of_heads',
                     'hidden_units',
                     'attention_type',
                     'aggregator_type',
                     'activate_function',
                     'number_of_heads']

categories = [macro_search_space[action] for action in macro_action_list]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Encode and embed GNN pipelines.')
    parser.add_argument('-a', '--algorithm', type=str,
                        default='t-SNE',
                        choices=['t-SNE', 'SVD', 'UMAP'],
                        help='Algorithms for embedding the one hot matrix.')
    parser.add_argument('-d', '--dimensions', type=int,
                        default=2,
                        help='Number of dimensions for embedding.')
    return parser.parse_args()


def main(args):
    archs = []
    # with open('macro_archs_full.txt', 'w') as f:
    #     for line in tqdm(f.readlines()):
    #         archs.append(list(literal_eval(line.strip())))
    for arch in product(macro_search_space['attention_type'],
                        macro_search_space['aggregator_type'],
                        macro_search_space['activate_function'],
                        macro_search_space['number_of_heads'],
                        macro_search_space['hidden_units'],
                        macro_search_space['attention_type'],
                        macro_search_space['aggregator_type'],
                        macro_search_space['activate_function'],
                        macro_search_space['number_of_heads']):
        # f.write(str(list(arch)) + '\n')
        # archs.append(str(list(arch)).replace('[', '').replace(']', '')
        #              .replace("'", '').replace(' ', ''))
        archs.append(list(arch))
    # pd.DataFrame(archs, columns=['arch']).reset_index().to_csv(
    #     'macro_archs_full_indexed.csv', index=False, sep=';')
    print(archs[0])
    # # archs = np.array(archs)
    enc = OneHotEncoder(handle_unknown='error', categories=categories)
    print('encoding...')
    X = enc.fit_transform(archs)
    print('encoding...DONE')
    print('Dimensions: ', X.shape)
    del archs
    print("Embedding on ", args.dimensions, 'dimensions.')
    if not exists('macro_full_one_hot.npz'):
        pickle.dump(enc, open('macro_full_one_hot_encoder.pkl', 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)
        scipy.sparse.save_npz('macro_full_one_hot.npz', X)
    if args.algorithm == 't-SNE':
        # Embed TSNE
        print('TSNE encoding...')
        X_embedded = TSNE(n_components=args.dimensions,
                          n_jobs=-1,
                          perplexity=20,
                          metric='hamming',
                          n_iter=250,
                          verbose=1000,
                          random_state=10).fit_transform(X.todense())
        print('TSNE encoding...DONE')
    elif args.algorithm == 'SVD':
        print('SVD encoding...')
        emb = TruncatedSVD(n_components=args.dimensions,
                           n_iter=10000,
                           random_state=10)
        X_embedded = emb.fit_transform(X.todense())
        print('SVD encoding...DONE')
        print('explained variance: ', emb.explained_variance_)
        print('explained variance ratio: ',
              np.sum(emb.explained_variance_ratio_))
        pickle.dump(emb, open('macro_full_SVD_encoder.pkl', 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)
    elif args.algorithm == 'UMAP':
        print('UMAP encoding...')
        emb = UMAP(n_neighbors=48,
                   n_components=args.dimensions,
                   # metric='hamming',
                   low_memory=True)
        X_embedded = emb.fit_transform(X)
        print('UMAP encoding...DONE')
    np.savetxt('macro_full_' + args.algorithm + '_embed.csv', X_embedded)


if __name__ == '__main__':
    args = parse_args()
    main(args)
