# Main imports
import networkx as nx
import numpy as np
from tqdm import tqdm
from node2vec import Node2Vec
from node2vec.edges import HadamardEmbedder

X = []
y = []
with open("./../Data/training.txt", "r") as f:
    for line in tqdm(f):
        line = line.split()
        X.append(np.array([int(line[0]), int(line[1])]))
        y.append(np.array(int(line[2])))
    X = np.array(X)
    y = np.array(y)

def fill_graph(X, y):
    G = nx.DiGraph()
    for nd, v in tqdm(zip(X, y)):
        if int(v) == 1:
            G.add_edge(nd[0], nd[1])
    return G

G= fill_graph(X, y)

node2vec = Node2Vec(G, dimensions=64, walk_length=9, num_walks=5, workers=100, p=1, q=2) 