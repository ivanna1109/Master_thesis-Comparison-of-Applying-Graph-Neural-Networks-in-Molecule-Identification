# Master_thesis-Comparison-of-Applying-Graph-Neural-Networks-in-Molecule-Identification

## Introduction
This repository contains the code and resources for my master's thesis, which focuses on comparing the application of Graph Neural Networks (GNNs) in molecule identification. The goal was to explore how different GNN architectures perform in predicting molecular activity, with a focus on their effectiveness in modeling molecular graphs and optimizing their hyperparameters.

## Problem Definition
Molecule identification is a critical task in computational biology and chemistry. This research addresses the challenge of predicting the biological activity of molecules by leveraging the structural information encoded in molecular graphs. Traditional methods often struggle with these complex structures, but Graph Neural Networks offer a promising approach for processing and learning from graph-based data.

## Molecular Graphs
In this work, molecules are represented as graphs, where atoms correspond to nodes, and chemical bonds correspond to edges. Molecules are modeled from their **SMILES** (Simplified Molecular Input Line Entry System) representation into graph structures, capturing essential chemical properties. In cases where node features are lacking or insufficient, methods like **DeepWalk** in combination with **Word2Vec** are employed to generate node embeddings, enriching the representation of molecular graphs. This approach enhances the model's ability to capture structural and contextual information within the graph.

## Models
Various GNN architectures have been explored, including:
- **Graph Convolutional Networks (GCN)**
- **GraphSAGE (Graph Sample and Agregation)**
- **Graph Attention Networks (GAT)**
- **Graph Isomorphism Networks (GIN)**
Each model is designed to capture different aspects of molecular structure, and their performance is evaluated based on their ability to accurately predict molecular activity.

## Optimization
Hyperparameter optimization plays a key role in improving model performance. In this study, the Optuna library is used to fine-tune model parameters such as learning rate, batch size, and network depth. The Bayesian optimization strategy provided by Optuna helps to efficiently explore the hyperparameter space, resulting in better model performance.

## Results
The results section includes detailed comparisons between the different GNN models in terms of classification accuracy, F1-score, and other relevant metrics. The models are evaluated on various datasets, and insights into their strengths and weaknesses in molecule identification are provided.

## Conclusion
This thesis demonstrates the effectiveness of GNNs in the task of molecule identification in case when molecular graphs are modeled based on SMILES representation. The models show promising results, and the insights gained from this research can help guide future developments in molecular prediction using graph-based machine learning models. Additionally, in future work, incorporating physicochemical features of atoms within molecules could further enhance model performance.
