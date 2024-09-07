import h5py
from spektral.data import Dataset, Graph
import numpy as np
import random

def create_graph_spectral(x, a, m_name):
    return Graph(x=x, a=a, molecule_name=m_name)

class MolecularDataset(Dataset):
    def __init__(self, file_path, **kwargs):
        self.file_path = file_path
        super().__init__(**kwargs)
    
    def read(self):
        graphs = []
        labels = []
        with h5py.File(self.file_path, 'r') as f:
            for i in range(len(f.keys())):
                grp = f[f'graph_{i}']
                
                x = grp['x'][:] #lista node featurea
                a = grp['a'][:] #matrica susedstva
                m_name = grp.attrs['molecule_name']
                y_label = grp['y'][()]#labela (numericka)
                graph = create_graph_spectral(x, a, m_name)
                graphs.append(graph)
                labels.append(y_label)
        
        return graphs, labels
    
def read_molecule_data():
    filename = "new_data_preprocess/augmented_graphs_new.h5"
    dataset = MolecularDataset(filename)
    graphs, labels = dataset.read()
    return graphs, labels

def read_molecule_data_oldF():
    filename = "new_data_preprocess/molecules_dataset_oldF.h5"
    dataset = MolecularDataset(filename)
    graphs, labels = dataset.read()
    return graphs, labels

def read_molecule_data_moreActive():
    filename = "new_data_preprocess/augmented_graphs_moreActive.h5"
    dataset = MolecularDataset(filename)
    graphs, labels = dataset.read()
    return graphs, labels