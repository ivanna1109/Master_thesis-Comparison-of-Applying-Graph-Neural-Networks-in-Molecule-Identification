
import networkx as nx
import numpy as np
from spektral.data import Graph, Dataset
import os
import h5py
from spektral.utils import one_hot
from imblearn.over_sampling import SMOTE
from gensim.models import Word2Vec
import random

def generate_random_walk(start_node, walk_length, graph):
    walk_sequence = [str(start_node)]

    for _ in range(walk_length):
        neighbors = list(graph.neighbors(start_node))
        if not neighbors:
            break  # Ako čvor nema susede, prekida šetnju
        next_node = np.random.choice(neighbors)
        walk_sequence.append(str(next_node))
        start_node = next_node

    return walk_sequence


def generate_random_walks(graph, num_walks, walk_length):
    walks = []

    for _ in range(num_walks):
        # Nasumično izabere početni čvor
        start_node = np.random.choice(list(graph.nodes()))
        # Generiše jednu šetnju
        walk = generate_random_walk(start_node, walk_length, graph)
        walks.append(walk)

    return walks

def deepwalk_embedding(graphs, dimensions=64):
    walks = []
    for G in graphs:
        walks.extend(list(generate_random_walks(G, num_walks=150, walk_length=35)))

    model = Word2Vec(sentences=walks, vector_size=dimensions, window=5, min_count=0, sg=1, workers=4)
    embeddings = [np.array([model.wv[str(node)] for node in G.nodes()]) for G in graphs]
    return embeddings

# 2. Dodavanje one-hot enkodirane atomske oznake cvorova u embeddinge
def add_element_features(graphs, embeddings, element_index):
    X_with_elements = []
    for G, embedding in zip(graphs, embeddings):
        element_features = np.zeros((len(G.nodes()), len(element_index)))
        for i, node in enumerate(G.nodes()):
            element = G.nodes[node].get('atom_type', '*')
            if element in element_index:
                element_features[i, element_index[element]] = 1
        combined_features = np.hstack([embedding, element_features])
        X_with_elements.append(combined_features)
    return X_with_elements

elements = ['O', 'C', 'Au', 'Yb', 'Ba', 'Mo', 'Pd', 'Hg', 
            'H', 'Nd', 'Dy', 'Se', 'Co', 'Li', 'Br', 'Bi', 'Cr', 'Ge',
              'Ag', 'Mg', 'Zr', 'Pt', 'Be', 'Ni', 'Cl', 'Gd', 'Sc', 'Tl', 
              'Zn', 'P', 'K', 'N', 'As', 'Ti', 'Sr', 'Sn', 'Na', 'Cd', 
              'Al', 'Cu', 'Si', 'F', 'Eu', 'Pb', 'Sb', 'V', 'S', 'Fe', 'B', 
              'I', 'Ca', 'In', 'Mn']  
element_index = {elem: idx for idx, elem in enumerate(elements)}
classes = {'inactive':0, 'partially active':1, 'active':2}
gml_files = [os.path.join('molecules', f) for f in os.listdir('molecules') if f.endswith('.gml')]
nx_graphs = []
labels = []
adj_matrixs = []

stop = False
for gml_file in gml_files:
    G = nx.read_gml(gml_file)
    G.graph['molecule_name'] = os.path.basename(gml_file)[9:-8]
    if all(deg >= 1 for node, deg in G.degree()):
        nx_graphs.append(G)
        labels.append(classes.get(G.graph.get('label')))
        original_matrix = nx.adjacency_matrix(G).toarray()
        adj_matrixs.append(original_matrix)
        if (G.graph.get('label')=='active' or G.graph.get('label')=='partially active'):
            range_gr = 4
            if (G.graph.get('label')=='active'): range_gr = 5
            for i in range(range_gr): 
                nodes = list(G.nodes)
                random.shuffle(nodes)
                mapping = {old_label: new_label for old_label, new_label in zip(G.nodes, nodes)}
                G_iso = nx.relabel_nodes(G, mapping)
                sorted_nodes = sorted(mapping.keys(), key=lambda x: mapping[x])
                sorted_indices = [list(G.nodes).index(node) for node in sorted_nodes]
                isomorphic_am = nx.adjacency_matrix(G_iso).toarray()
                A_sorted = isomorphic_am[np.ix_(sorted_indices, sorted_indices)]
                new_G_iso = nx.from_numpy_array(A_sorted)
                node_labels = nx.get_node_attributes(G_iso, 'atom_label')

                inverse_mapping = {v: k for k, v in mapping.items()}
                for idx, node in enumerate(sorted_indices):
                    print(node_labels[str(node)])
                    new_G_iso.nodes[node]['atom_label'] = node_labels[str(node)]
                new_G_iso.graph['molecule_name'] = G.graph['molecule_name']
                nx_graphs.append(new_G_iso)
                labels.append(classes.get(G.graph.get('label')))
                adj_matrixs.append(A_sorted)

print("Ucitani svi grafovi iz foldera")
print(f"Num graphs: {len(nx_graphs)}")
print(f"Num labels: {len(labels)}")
print(nx_graphs[0].graph.get('molecule_name'))
print(labels[0])
# Brojanje elemenata
count_0 = labels.count(0)
count_1 = labels.count(1)
count_2 = labels.count(2)

print(f"Broj elemenata sa oznakom 0: {count_0}")
print(f"Broj elemenata sa oznakom 1: {count_1}")
print(f"Broj elemenata sa oznakom 2: {count_2}")


X_embeddings = deepwalk_embedding(nx_graphs)

X_embeddings_with_elements = add_element_features(nx_graphs, X_embeddings, element_index)

with h5py.File('new_data_preprocess/augmented_graphs_moreActive.h5', 'w') as f:
    for i, graph in enumerate(nx_graphs):
        grp = f.create_group(f'graph_{i}')
        grp.create_dataset('x', data=X_embeddings_with_elements[i])
        grp.create_dataset('a', data=adj_matrixs[i])
        grp.create_dataset('y', data=labels[i])

        grp.attrs['molecule_name'] = graph.graph.get('molecule_name')

print(X_embeddings_with_elements[0].shape)
print(adj_matrixs[0])
print(labels[0])
print("Podaci su uspešno sačuvani u 'augmented_graphs_moreActive.h5'")

