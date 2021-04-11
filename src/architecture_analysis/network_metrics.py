import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
import itertools as it
from os.path import join
from os.path import isfile
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from scipy.sparse import load_npz
from scipy.spatial.distance import hamming

matplotlib.use('Agg')  # fix ioctl error 25
sns.set()

paths = {
    'gecco_dir': '/scratch/math/backup_gecco/',
    'dist_dir': '/scratch/math/backup_gecco/dist'
}

n_nodes = 89820

G = nx.Graph()
# Add all nodes
G.add_nodes_from(range(n_nodes))

# Import edges
with open(join(paths['dist_dir'], 'neigh_all.txt'), 'r') as f:
    for line in f.readlines():
        u, v, _ = line.split()
        G.add_edge(int(u), int(v))

print('Nodes: ', G.number_of_nodes(), 'Edges: ', G.number_of_edges())

# Connected components information
comp = [c for c in sorted(nx.connected_components(G), key=len, reverse=True)]
print("# of components ", len(comp))
sizecomponents = [len(c) for c in comp]
# print("Size of components: ", sizecomponents)

# Load unique archs
unique_macro_archs = pd.read_csv(
    'unique_archs_without_last_dim.txt',
    header=None,
    sep=';',
    names=['arch']).reset_index()

# Cora
cora_results = pd.read_csv(
    '/scratch/math/backup_gecco/Cora/Cora_all_results_train.csv',
    sep=';')
cora_indexed = pd.merge(unique_macro_archs,
                        cora_results[['arch', 'val_acc']],
                        on=['arch'])
cora_indexed.head()
cora_small = pd.read_csv('cora_reduced_space.csv', sep=';')
# Select archs and indexes
cora_small_indexed = pd.merge(cora_small['arch'],
                              unique_macro_archs, on=['arch'])
# Get validation scores
cora_small_indexed = pd.merge(cora_small_indexed.drop(columns=['arch']),
                              cora_indexed, on=['index'])

# Citeseer
citeseer_results = pd.read_csv(
    '/scratch/math/backup_gecco/Citeseer/Citeseer_300_10_train_unified.csv',
    sep=';')
citeseer_indexed = pd.merge(unique_macro_archs,
                            citeseer_results[['arch', 'val_acc']],
                            on=['arch'])
citeseer_small = pd.read_csv('citeseer_reduced_space.csv', sep=';')
# Grab indexes
cit_small_indexed = pd.merge(citeseer_small['arch'],
                             unique_macro_archs, on=['arch'])
# Grab validation accuracies
cit_small_indexed = pd.merge(cit_small_indexed.drop(columns=['arch']),
                             citeseer_indexed, on=['index'])

# Pubmed
pubmed_results = pd.read_csv(
    '/scratch/math/backup_gecco/Pubmed/Pubmed_current_full_train.csv',
    sep=';')
pubmed_indexed = pd.merge(unique_macro_archs,
                          pubmed_results[['arch', 'val_acc']],
                          on=['arch'])
pubmed_small = pd.read_csv('pubmed_reduced_space.csv', sep=';')
# Grab indexes
pub_small_indexed = pd.merge(pubmed_small['arch'],
                             unique_macro_archs, on=['arch'])
# Grab validation accuracies
pub_small_indexed = pd.merge(pub_small_indexed.drop(columns=['arch']),
                             pubmed_indexed, on=['index'])

# Load one hot encoded vectors
macro_one_hot_archs = load_npz('one_hot_encoded_unique_macro_archs.npz')
macro_one_hot_archs = macro_one_hot_archs.todense()
n_nodes, vdim = macro_one_hot_archs.shape


# FLA metrics function
# Calculate neutrality ratio
def calculate_neutrality(G, indexed_scores):
    deg = []
    ratio = []
    for node in indexed_scores['index']:
        node_acc = indexed_scores[
            indexed_scores['index'] == node]['val_acc'].values[0]
        # Get neighbor accs
        neigh_accs = indexed_scores[
            indexed_scores['index'].isin(G[node])]['val_acc']
        n_neighs = neigh_accs.shape[0]
        neut_deg = np.where(neigh_accs == node_acc)[0].shape[0]
        deg.append(neut_deg)
        ratio.append(neut_deg / n_neighs if n_neighs > 0 else 0.0)
    return pd.Series(deg), pd.Series(ratio)


def calculate_round_neutrality(G, indexed_scores, r=3):
    assert(r > 0)
    deg = []
    ratio = []
    tol = 1e-6
    for node in indexed_scores['index']:
        if G.degree()[node] == 0:
            continue
        node_acc = indexed_scores[
            indexed_scores['index'] == node]['val_acc'].values[0]
        # Get neighbor accs
        neigh_accs = indexed_scores[
            indexed_scores['index'].isin(G[node])]['val_acc']
        neigh_accs = np.round(neigh_accs, r)
        node_acc = np.round(node_acc, r)
        n_neighs = neigh_accs.shape[0]
        # Calculate absolute difference in fitness to neighbors
        diffs = np.absolute(neigh_accs - node_acc)
        # Get all those that the difference is smaller than the tolerance
        neut_deg = np.where(diffs < tol)[0].shape[0]
        deg.append(neut_deg)
        ratio.append(neut_deg / n_neighs if n_neighs > 0 else 0.0)
    return pd.Series(deg), pd.Series(ratio)


# FDC
# Discover the best performing arch
def FDC(indexed_scores, macro_one_hot_archs, vdim):
    best_index = indexed_scores.iloc[
        indexed_scores['val_acc'].idxmax()]['index']
    dist_to_best = np.array([hamming(macro_one_hot_archs[best_index],
                                     macro_one_hot_archs[index]) * vdim
                             for index in indexed_scores['index']])
    g = sns.regplot(dist_to_best, indexed_scores['val_acc'].values)
    g.set(xticks=[x for x in range(0, 20, 2)])
    c = np.cov(indexed_scores['val_acc'].values, dist_to_best)[0][1]
    fdc = c / (np.std(dist_to_best) * np.std(indexed_scores['val_acc'].values))
    return fdc


# Dispersion
def dispersion(idx_scores, one_hot, sample_size=1000, frac=0.1, seed=1):
    # Take sample of dataset
    s_prime = idx_scores.sample(n=sample_size,
                                random_state=seed).reset_index()
    # Take one-hot encoded vectors in this sample
    s_prime_vec = one_hot[s_prime['index']]
    # Pair-wise distance
    avg_dist_s_prime = [hamming(s_prime_vec[x],
                                s_prime_vec[y])
                        for x in range(s_prime_vec.shape[0])
                        for y in range(x, s_prime_vec.shape[0])]
    # Avg. pairwise distance
    avg_dist_s_prime = np.mean(avg_dist_s_prime)
    # Sort by accuracy
    sorted_idx_scores = s_prime.sort_values(by=['val_acc'])
    # Take the better x% fraction
    s_star = sorted_idx_scores[int((sample_size * -1) * frac):]
    # Take one-hot encoded vectors from this parcel
    s_star_vec = one_hot[s_star['index']]
    # Pairwise distance
    avg_dist_s_star = [hamming(s_star_vec[x],
                               s_star_vec[y])
                       for x in range(s_star_vec.shape[0])
                       for y in range(x, s_star_vec.shape[0])]
    avg_dist_s_star = np.mean(avg_dist_s_star)
    return avg_dist_s_star - avg_dist_s_prime


def calculate_FLA_graph(G, comp):
    dfs = {}
    list_df_names = [('cora_small_comp_stats', cora_small_indexed),
                     ('cora_full_comp_stats', cora_indexed),
                     ('cit_small_comp_stats', cit_small_indexed),
                     ('cit_full_comp_stats', citeseer_indexed),
                     ('pub_small_comp_stats', pub_small_indexed),
                     ('pub_full_comp_stats', pubmed_indexed)]
    # Build components df
    for df_name, df in tqdm(list_df_names):
        # Create result DF
        dfs[df_name] = pd.DataFrame(
            columns=['id',
                     '#archs',
                     'avg',
                     'median',
                     'std',
                     'best_arch',
                     'best_arch_fitness',
                     'worst_arch',
                     'worst_arch_fitness',
                     'avg. neutrality_ratio',
                     'median. neutrality_ratio',
                     'std. neutrality_ratio',
                     'Q1 neutrality_ratio',
                     'Q3 neutrality_ratio',
                     'min neutrality_ratio',
                     'max neutrality_ratio'])
        for idx, c in tqdm(enumerate(comp)):
            try:
                # Select subgraph:
                subgraph = G.subgraph(c).copy()
                # Select nodes
                selected_comp = df[df['index'].isin(list(subgraph.nodes()))]
                if selected_comp.shape[0] <= 1:
                    continue
                results = {}
                results['id'] = idx
                results['#archs'] = selected_comp.shape[0]
                results['avg'] = np.mean(selected_comp['val_acc'].values)
                results['median'] = np.median(selected_comp['val_acc'].values)
                results['std'] = np.std(selected_comp['val_acc'].values)
                _, neut_ratio_df = calculate_round_neutrality(subgraph,
                                                              selected_comp,
                                                              3)
                if neut_ratio_df.shape[0] > 0:
                    results['avg. neutrality_ratio'] = np.mean(neut_ratio_df)
                    results['median. neutrality_ratio'] = \
                        np.median(neut_ratio_df)
                    results['std. neutrality_ratio'] = np.std(neut_ratio_df)
                    results['Q1 neutrality_ratio'] = \
                        np.percentile(neut_ratio_df, 25)
                    results['Q3 neutrality_ratio'] = \
                        np.percentile(neut_ratio_df, 75)
                    results['min neutrality_ratio'] = np.min(neut_ratio_df)
                    results['max neutrality_ratio'] = np.max(neut_ratio_df)
                else:
                    results['avg. neutrality_ratio'] = 0.0
                    results['median. neutrality_ratio'] = 0.0
                    results['std. neutrality_ratio'] = 0.0
                    results['Q1 neutrality_ratio'] = 0.0
                    results['Q3 neutrality_ratio'] = 0.0
                    results['min neutrality_ratio'] = 0.0
                    results['max neutrality_ratio'] = 0.0
                selected_comp = selected_comp.reset_index(drop=True)
                best = selected_comp.iloc[selected_comp['val_acc'].idxmax()]
                # print(best)
                results['best_arch'] = best['arch']
                results['best_arch_fitness'] = best['val_acc']
                worst = selected_comp.iloc[selected_comp['val_acc'].idxmin()]
                results['worst_arch'] = worst['arch']
                results['worst_arch_fitness'] = worst['val_acc']
                results = pd.DataFrame(results, index=[0])
                dfs[df_name] = pd.concat([dfs[df_name],
                                          results],
                                         axis=0,
                                         ignore_index=True)
            except Exception as e:
                print(results)
                raise e
        # Save DF
        dfs[df_name].to_csv(df_name + '.csv', index=False, sep=';')


calculate_FLA_graph(G, comp)

degrees = list(dict(G.degree()).values())
unique_degrees_count = len(np.unique(degrees))

print("DEGREE INFORMATION\nTotal")
print("Range: ", [min(degrees), max(degrees)])
print("Mean: ", np.average(degrees))
print("Std.: ", np.sqrt(np.var(degrees)))
print("Median: ", np.median(degrees))
print("Q1: ", np.percentile(degrees, 25))
print("Q3: ", np.percentile(degrees, 75))

# Results total
plt.figure(figsize=(20, 5))
plt.subplot(1, 3, 1)
g1 = plt.hist(degrees, rwidth=0.8, bins=unique_degrees_count, color='hotpink')
lx1 = plt.xlabel("degree total")
ly1 = plt.ylabel("#nodes")
# Y in log scale
plt.subplot(1, 3, 2)
g1a = plt.hist(degrees, rwidth=0.8, bins=unique_degrees_count, color='hotpink')
lx1a = plt.xlabel("degree total")
yscale1a = plt.yscale('log')
# X and Y in log scale
plt.subplot(1, 3, 3)
g1a = plt.hist(degrees, rwidth=0.8, bins=unique_degrees_count, color='hotpink')
lx1a = plt.xlabel("degree total")
yscale1b = plt.yscale('log')
xscale1b = plt.xscale('log')
plt.tight_layout()
plt.savefig(join(paths['dist_dir'], 'degree_plots.pdf'))

# Studying the largest connected component now
LCC = max(nx.connected_component_subgraphs(G, copy=True), key=len)
print("#nodes: ", LCC.number_of_nodes())
print("#edges: ", LCC.number_of_edges())

# Eccentricity
# The eccentricity of a node $v$ is
# the maximum distance from $v$ to all other nodes in $G$.

ecc_filename = join(paths['dist_dir'], 'neigh_largest_lcc_ecc.p')
# If file does not exist yet, create it
if not isfile(ecc_filename):
    print('calculating ecc total...')
    ecc = nx.eccentricity(LCC)
    print('calculating ecc total...DONE')
    print('saving eccentricity total dict')
    pickle.dump(ecc, open(ecc_filename, 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)
    print('saved ecc dict')
else:
    ecc = pickle.load(open(ecc_filename, 'rb'))

eccentricity = list(ecc.values())
print("ECCENTRICITY INFORMATION")
print("Range: ", [min(eccentricity), max(eccentricity)])
print("Mean: ", np.average(eccentricity))
print("Std.: ", np.sqrt(np.var(eccentricity)))
print("Median: ", np.median(eccentricity))
print("Q1: ", np.percentile(eccentricity, 25))
print("Q3: ", np.percentile(eccentricity, 75))

# Diameter -- maximum eccentricity
print("size_diameter: ", nx.diameter(LCC, e=ecc))
# Plot eccentricities
plt.figure(figsize=(10, 5))
plt.hist(eccentricity, rwidth=0.8, bins=unique_degrees_count,
         color='midnightblue')
plt.xlabel("eccentricity total")
plt.ylabel("#nodes")
plt.tight_layout()
plt.savefig(join(paths['dist_dir'], 'eccentricity_plot.pdf'))

# Triangle metrics
# A triangle in a grapgh $G = (V, E)$ is formed
# by three nodes $(u, v, w)$
# when $(u, v), (v, w), (u, w) \in E$.

number_triangles = list(nx.triangles(G).values())

print("Triangles in Whole Graph")
print("Range: ", [min(number_triangles), max(number_triangles)])
print("Mean: ", np.average(number_triangles))
print("Std.: ", np.sqrt(np.var(number_triangles)))
print("Median: ", np.median(number_triangles))
print("Q1: ", np.percentile(number_triangles, 25))
print("Q3: ", np.percentile(number_triangles, 75))


number_triangles_LCC = list(nx.triangles(LCC).values())

print("Triangles in LCC")
print("Range: ", [min(number_triangles_LCC), max(number_triangles_LCC)])
print("Mean: ", np.average(number_triangles_LCC))
print("Std.: ", np.sqrt(np.var(number_triangles_LCC)))
print("Median: ", np.median(number_triangles_LCC))
print("Q1: ", np.percentile(number_triangles_LCC, 25))
print("Q3: ", np.percentile(number_triangles_LCC, 75))

# Transitivity
# The fraction of existing triangles over all possible triangles present in G
fractions_triangles_total = nx.transitivity(G)
fractions_triangles_LCC = nx.transitivity(LCC)
print('Transitivity')
print("Whole Graph: ", fractions_triangles_total)
print("LCC: ", fractions_triangles_LCC)
# Results
frequency = [len(list(group))
             for key, group in it.groupby(sorted(number_triangles))]
frequency_LCC = [len(list(group))
                 for key, group in it.groupby(sorted(number_triangles_LCC))]

# Plot triangles
plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
# Plot graph triangles
g5 = plt.stem(np.array(np.unique(number_triangles)).tolist(),
              frequency, linefmt='C9-',
              markerfmt='C9o', basefmt='C4--')
lx5 = plt.xlabel("#triangles Whole Graph")
ly5 = plt.ylabel("#nodes")
xscale5 = plt.xscale('log')
# Plot LCC triangles
plt.subplot(1, 2, 2)
g5 = plt.stem(np.array(np.unique(number_triangles_LCC)).tolist(),
              frequency, linefmt='C9-',
              markerfmt='C9o', basefmt='C4--')
lx5 = plt.xlabel("#triangles LCC")
ly5 = plt.ylabel("#nodes")
xscale5 = plt.xscale('log')
plt.tight_layout()
plt.savefig(join(paths['dist_dir'], 'triangle_plot.pdf'))

# Clustering coefficient
clus = list(nx.clustering(G).values())
clus_LCC = list(nx.clustering(LCC).values())

print("Clustering total")
print("Range: ", [min(clus), max(clus)])
print("Mean: ", np.average(clus))
print("Std.: ", np.sqrt(np.var(clus)))
print("Median: ", np.median(clus))
print("Q1: ", np.percentile(clus, 25))
print("Q3: ", np.percentile(clus, 75))


print("Clustering LCC")
print("Range: ", [min(clus_LCC), max(clus_LCC)])
print("Mean: ", np.average(clus_LCC))
print("Std.: ", np.sqrt(np.var(clus_LCC)))
print("Median: ", np.median(clus_LCC))
print("Q1: ", np.percentile(clus_LCC, 25))
print("Q3: ", np.percentile(clus_LCC, 75))

# Results
plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
g6 = plt.hist(clus, rwidth=0.8, bins=unique_degrees_count, color='c')
lx6 = plt.xlabel("Clustering coefficient Whole Graph")
ly6 = plt.ylabel("#nodes")
yscale6 = plt.yscale('log')
plt.subplot(1, 2, 2)
g6b = plt.hist(clus_LCC, rwidth=0.8, bins=unique_degrees_count, color='c')
lx6b = plt.xlabel("Clustering coefficient LCC")
yscale6b = plt.yscale('log')
plt.tight_layout()
plt.savefig(join(paths['dist_dir'], 'clustering_coefficient_plots.pdf'))

# Centrality Metrics
# Betweenness metrics

node_bet_filename = join(paths['dist_dir'], 'node_bet_whole.p')
# If file does not exist yet, create it
if not isfile(node_bet_filename):
    print('calculating node betweenness total...')
    nodes_betweenness = nx.betweenness_centrality(G)
    print('calculating node betweenness total...DONE')
    print('saving node betweenness total dict')
    pickle.dump(nodes_betweenness, open(node_bet_filename, 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)
    print('saved ecc dict')
else:
    print('loading node betweenness total from file')
    nodes_betweenness = pickle.load(open(node_bet_filename, 'rb'))


node_bet_lcc_filename = join(paths['dist_dir'], 'node_bet_lcc.p')
# If file does not exist yet, create it
if not isfile(node_bet_lcc_filename):
    print('calculating node betweenness total...')
    nodes_betweenness_LCC = nx.betweenness_centrality(LCC)
    print('calculating node betweenness total...DONE')
    print('saving node betweenness total dict')
    pickle.dump(nodes_betweenness_LCC,
                open(node_bet_lcc_filename, 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)
    print('saved ecc dict')
else:
    print('loading node betweenness total from file')
    nodes_betweenness_LCC = pickle.load(
        open(node_bet_lcc_filename, 'rb'))

print('Node Betweenness Centrality')
bet_total = list(nodes_betweenness.values())
bet_total_LCC = list(nodes_betweenness_LCC.values())

print("Betweenness centrality Whole Graph")
print("Range: ", [min(bet_total), max(bet_total)])
print("Mean: ", np.average(bet_total))
print("Std.: ", np.sqrt(np.var(bet_total)))
print("Median: ", np.median(bet_total))
print("Q1: ", np.percentile(bet_total, 25))
print("Q3: ", np.percentile(bet_total, 75))

print("\nBetweenness centrality LCC")
print("Range: ", [min(bet_total_LCC), max(bet_total_LCC)])
print("Mean: ", np.average(bet_total_LCC))
print("Std.: ", np.sqrt(np.var(bet_total_LCC)))
print("Median: ", np.median(bet_total_LCC))
print("Q1: ", np.percentile(bet_total_LCC, 25))
print("Q3: ", np.percentile(bet_total_LCC, 75))

# Results
plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
g7 = plt.hist(bet_total, rwidth=0.8, bins=unique_degrees_count, color='r')
lx7 = plt.xlabel("Betweenness Whole Graph")
ly7 = plt.ylabel("#nodes")
yscale7 = plt.yscale('log')
plt.subplot(1, 2, 2)
g7a = plt.hist(bet_total_LCC, rwidth=0.8, bins=unique_degrees_count, color='r')
lx7a = plt.xlabel("Betweenness LCC")
yscale7a = plt.yscale('log')
plt.tight_layout()
plt.savefig(join(paths['dist_dir'], 'node_betweenness_plots.pdf'))


# Edge Betweenness
edges_bet = list(nx.edge_betweenness_centrality(G).values())
edges_bet_LCC = list(nx.edge_betweenness_centrality(LCC).values())

print('\nEdge Betweenness Whole Graph:')
print("Range: ", [min(edges_bet), max(edges_bet)])
print("Mean: ", np.average(edges_bet))
print("Std.: ", np.sqrt(np.var(edges_bet)))
print("Median: ", np.median(edges_bet))
print("Q1: ", np.percentile(edges_bet, 25))
print("Q3: ", np.percentile(edges_bet, 75))

print('\nEdge Betweenness LCC:')
print("Range: ", [min(edges_bet_LCC), max(edges_bet_LCC)])
print("Mean: ", np.average(edges_bet_LCC))
print("Std.: ", np.sqrt(np.var(edges_bet_LCC)))
print("Median: ", np.median(edges_bet_LCC))
print("Q1: ", np.percentile(edges_bet_LCC, 25))
print("Q3: ", np.percentile(edges_bet_LCC, 75))

plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
g8 = plt.hist(edges_bet, rwidth=0.8, bins=unique_degrees_count, color='r')
lx8 = plt.xlabel("Edges Betweenness")
ly8 = plt.ylabel("#nodes")

plt.subplot(1, 2, 2)
g8a = plt.hist(edges_bet_LCC, rwidth=0.8, bins=unique_degrees_count, color='r')
lx8a = plt.xlabel("Edges Betweenness")
yscale8a = plt.yscale('log')
plt.tight_layout()
plt.savefig(join(paths['dist_dir'], 'edge_betweenness_plots.pdf'))


# Closeness centrality
# Closeness of a node u is the reciprocal of
# the sum of the shortest path distances
# from u to all n-1 other nodes.
print("\n\nCloseness Centrality")
close = nx.closeness_centrality(G)
close_total = list(close.values())
close = nx.closeness_centrality(LCC)
close_total_LCC = list(close.values())


print("\nCloseness total")
print("Range: ", [min(close_total), max(close_total)])
print("Mean: ", np.average(close_total))
print("Std.: ", np.sqrt(np.var(close_total)))
print("Median: ", np.median(close_total))
print("Q1: ", np.percentile(close_total, 25))
print("Q3: ", np.percentile(close_total, 75))

print("\nCloseness LCC")
print("Range: ", [min(close_total_LCC), max(close_total_LCC)])
print("Mean: ", np.average(close_total_LCC))
print("Std.: ", np.sqrt(np.var(close_total_LCC)))
print("Median: ", np.median(close_total_LCC))
print("Q1: ", np.percentile(close_total_LCC, 25))
print("Q3: ", np.percentile(close_total_LCC, 75))

# Results
plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
g9 = plt.hist(close_total, rwidth=0.8,
              bins=unique_degrees_count, color='goldenrod')
lx9 = plt.xlabel("Closeness centrality total")
ly9 = plt.ylabel("#nodes")

plt.subplot(1, 2, 2)
g9a = plt.hist(close_total_LCC, rwidth=0.8,
               bins=unique_degrees_count, color='goldenrod')
lx9a = plt.xlabel("Closeness centrality LCC")

plt.tight_layout()
plt.savefig(join(paths['dist_dir'], 'closeness_plots.pdf'))

# Load centrality --
# fraction of all shortest paths that pass through a node.
print("\n\nLoad centrality")
centrality = nx.load_centrality(G)
centrality_total = list(centrality.values())
centrality = nx.load_centrality(LCC)
centrality_total_LCC = list(centrality.values())

print("\nLoad centrality Whole Graph")
print("Range: ", [min(centrality_total), max(centrality_total)])
print("Mean: ", np.average(centrality_total))
print("Std.: ", np.sqrt(np.var(centrality_total)))
print("Median: ", np.median(centrality_total))
print("Q1: ", np.percentile(centrality_total, 25))
print("Q3: ", np.percentile(centrality_total, 75))

print("\nLoad centrality LCC")
print("Range: ", [min(centrality_total_LCC), max(centrality_total_LCC)])
print("Mean: ", np.average(centrality_total_LCC))
print("Std.: ", np.sqrt(np.var(centrality_total_LCC)))
print("Median: ", np.median(centrality_total_LCC))
print("Q1: ", np.percentile(centrality_total_LCC, 25))
print("Q3: ", np.percentile(centrality_total_LCC, 75))

# Results
plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
g10 = plt.hist(centrality_total, rwidth=0.8,
               bins=unique_degrees_count, color='goldenrod')
lx10 = plt.xlabel("Load centrality total")
ly10 = plt.ylabel("#nodes")
yscale10 = plt.yscale('log')
plt.subplot(1, 2, 2)
g10a = plt.hist(centrality_total_LCC, rwidth=0.8,
                bins=unique_degrees_count, color='goldenrod')
lx10a = plt.xlabel("Load centrality LCC")
yscale10a = plt.yscale('log')

plt.tight_layout()
plt.savefig(join(paths['dist_dir'], 'load_centrality_plots.pdf'))


# Harmonic centrality --
# Reverses the sum and reciprocal operations
# in the definition of closeness centrality
print("\n\nHarmonic centrality")
harmonic = nx.harmonic_centrality(G)
harmonic_total = list(harmonic.values())
harmonic_LCC = nx.harmonic_centrality(LCC)
harmonic_total_LCC = list(harmonic_LCC.values())

print("\nHarmonic centrality total")
print("Range: ", [min(harmonic_total), max(harmonic_total)])
print("Mean: ", np.average(harmonic_total))
print("Std.: ", np.sqrt(np.var(harmonic_total)))
print("Median: ", np.median(harmonic_total))
print("Q1: ", np.percentile(harmonic_total, 25))
print("Q3: ", np.percentile(harmonic_total, 75))

print("\nHarmonic centrality target")
print("Range: ", [min(harmonic_total_LCC), max(harmonic_total_LCC)])
print("Mean: ", np.average(harmonic_total_LCC))
print("Std.: ", np.sqrt(np.var(harmonic_total_LCC)))
print("Median: ", np.median(harmonic_total_LCC))
print("Q1: ", np.percentile(harmonic_total_LCC, 25))
print("Q3: ", np.percentile(harmonic_total_LCC, 75))

# Results
plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
g11 = plt.hist(harmonic_total, rwidth=0.8,
               bins=unique_degrees_count, color='goldenrod')
lx11 = plt.xlabel("Harmonic centrality total")
ly11 = plt.ylabel("#nodes")
yscale11 = plt.yscale('log')

plt.subplot(1, 2, 2)
g11a = plt.hist(harmonic_total_LCC, rwidth=0.8,
                bins=unique_degrees_count, color='goldenrod')
lx11a = plt.xlabel("Harmonic centrality LCC")

plt.tight_layout()
plt.savefig(join(paths['dist_dir'], 'harmonic_centrality_plots.pdf'))

# Degree centrality --
# Measures the number of direct neighbors
print("\n\nDegree centrality")
degree_cent = nx.degree_centrality(G)
degree_cent_total = list(degree_cent.values())
degree_cent = nx.degree_centrality(LCC)
degree_cent_total_LCC = list(degree_cent.values())

print("Degree centrality total")
print("Range: ", [min(degree_cent_total), max(degree_cent_total)])
print("Mean: ", np.average(degree_cent_total))
print("Std.: ", np.sqrt(np.var(degree_cent_total)))
print("Median: ", np.median(degree_cent_total))
print("Q1: ", np.percentile(degree_cent_total, 25))
print("Q3: ", np.percentile(degree_cent_total, 75))

print("\nDegree centrality LCC")
print("Range: ", [min(degree_cent_total_LCC),
                  max(degree_cent_total_LCC)])
print("Mean: ", np.average(degree_cent_total_LCC))
print("Std.: ", np.sqrt(np.var(degree_cent_total_LCC)))
print("Median: ", np.median(degree_cent_total_LCC))
print("Q1: ", np.percentile(degree_cent_total_LCC, 25))
print("Q3: ", np.percentile(degree_cent_total_LCC, 75))

# Results
plt.figure(figsize=(20, 5))
plt.subplot(1, 3, 1)
g12 = plt.hist(degree_cent_total, rwidth=0.8,
               bins=unique_degrees_count, color='goldenrod')
lx12 = plt.xlabel("Degree centrality total")
ly12 = plt.ylabel("#nodes")
yscale12 = plt.yscale('log')

plt.subplot(1, 2, 2)
g12a = plt.hist(degree_cent_total_LCC, rwidth=0.8,
                bins=unique_degrees_count, color='goldenrod')
lx12a = plt.xlabel("Degree centrality LCC")
plt.tight_layout()
plt.savefig(join(paths['dist_dir'], 'degree_centrality_plots.pdf'))


# Eigenvector centrality --
# Measures the influence of a node in a network.
print("\n\nEigenvector centrality")
eigen_cent = nx.eigenvector_centrality(G)
eigen_cent_total = list(eigen_cent.values())
eigen_cent = nx.eigenvector_centrality(LCC)
eigen_cent_LCC = list(eigen_cent.values())


print("Eigenvector centrality total")
print("Range: ", [min(eigen_cent_total), max(eigen_cent_total)])
print("Mean: ", np.average(eigen_cent_total))
print("Std.: ", np.sqrt(np.var(eigen_cent_total)))
print("Median: ", np.median(eigen_cent_total))
print("Q1: ", np.percentile(eigen_cent_total, 25))
print("Q3: ", np.percentile(eigen_cent_total, 75))

print("\nEigenvector centrality LCC")
print("Range: ", [min(eigen_cent_LCC), max(eigen_cent_LCC)])
print("Mean: ", np.average(eigen_cent_LCC))
print("Std.: ", np.sqrt(np.var(eigen_cent_LCC)))
print("Median: ", np.median(eigen_cent_LCC))
print("Q1: ", np.percentile(eigen_cent_LCC, 25))
print("Q3: ", np.percentile(eigen_cent_LCC, 75))

# Results
plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
g13 = plt.hist(eigen_cent_total, rwidth=0.8,
               bins=unique_degrees_count, color='goldenrod')
lx13 = plt.xlabel("Eigenvector centrality total")
ly13 = plt.ylabel("#nodes")
yscale13 = plt.yscale('log')

plt.subplot(1, 2, 2)
g13a = plt.hist(eigen_cent_LCC, rwidth=0.8,
                bins=unique_degrees_count, color='goldenrod')
lx13a = plt.xlabel("Eigenvector centrality LCC")
plt.tight_layout()
plt.savefig(join(paths['dist_dir'], 'eigenvector_centrality_plots.pdf'))

# Assortativity -- Homophily metrics
# Assortativity measures the similarity of
# connections in the graph with respect to the node degree.

# Assortativity coefficient
# -- Measures the preference for a network's nodes
# to attach to others that are similar in some way
# Computes degree assortativity of graph.
print("\n\nAssortativity coefficient ")
degree_assort_total = nx.degree_assortativity_coefficient(G)
print("Assortativity coefficient - Total: ", degree_assort_total)
degree_assort_target_LCC = nx.degree_assortativity_coefficient(LCC)
print("Assortativity coefficient - LCC: ", degree_assort_target_LCC)

# Average neighbor degree
averageneighbor = nx.average_neighbor_degree(G)
averageneighbor_total = list(averageneighbor.values())
averageneighbor = nx.average_neighbor_degree(LCC)
averageneighbor_LCC = list(averageneighbor.values())

print("Average neighbor degree total")
print("Range: ", [min(averageneighbor_total), max(averageneighbor_total)])
print("Mean: ", np.average(averageneighbor_total))
print("Std.: ", np.sqrt(np.var(averageneighbor_total)))
print("Median: ", np.median(averageneighbor_total))
print("Q1: ", np.percentile(averageneighbor_total, 25))
print("Q3: ", np.percentile(averageneighbor_total, 75))

print("\nAverage neighbor degree LCC")
print("Range: ", [min(averageneighbor_LCC),
                  max(averageneighbor_LCC)])
print("Mean: ", np.average(averageneighbor_LCC))
print("Std.: ", np.sqrt(np.var(averageneighbor_LCC)))
print("Median: ", np.median(averageneighbor_LCC))
print("Q1: ", np.percentile(averageneighbor_LCC, 25))
print("Q3: ", np.percentile(averageneighbor_LCC, 75))

# Results
plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
g14 = plt.hist(averageneighbor_total, rwidth=0.8,
               bins=unique_degrees_count, color='purple')
lx14 = plt.xlabel("Average neighbor degree total")
ly14 = plt.ylabel("#nodes")
yscale14 = plt.yscale('log')
xscale14 = plt.xscale('log')
plt.subplot(1, 2, 2)
g14a = plt.hist(averageneighbor_LCC, rwidth=0.8,
                bins=unique_degrees_count, color='purple')
lx14a = plt.xlabel("Average neighbor degree target")
yscale14a = plt.yscale('log')
xscale14a = plt.xscale('log')
plt.tight_layout()
plt.savefig(join(paths['dist_dir'], 'average_neighbors_plots.pdf'))
