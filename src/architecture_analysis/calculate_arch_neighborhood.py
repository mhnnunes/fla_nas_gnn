
from scipy.sparse import load_npz
from scipy.spatial import distance
from joblib import Parallel
from joblib import delayed


def chunks(lst, n):
    """
    Yield successive n-sized chunks from lst.
    from:
    https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def _proc(x, y):
    global macro_one_hot_archs
    global f
    d = distance.hamming(macro_one_hot_archs[x],
                         macro_one_hot_archs[y])
    d *= 32  # 64 / 2
    f.write(' '.join([str(x), str(y), str(d)]) + '\n')


# Import one-hot encoded architectures
macro_one_hot_archs = load_npz('one_hot_encoded_unique_macro_archs.npz')
macro_one_hot_archs = macro_one_hot_archs.todense()
print('Read archs')
# Create neighborhood graph
n_nodes = macro_one_hot_archs.shape[0]
print('nodes: ', n_nodes)


f = open('distances2.txt', 'w')
dists = \
    Parallel(n_jobs=-1,
             verbose=10,
             batch_size=32768,
             backend='multiprocessing')(delayed(_proc)(x, y)
                                        for x in range(int(n_nodes // 2))
                                        for y in range(n_nodes)
                                        if x < y)
print('calculated dists')

print("END!")
