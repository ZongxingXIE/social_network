"""
In this script, we provide two network generation models:
1) func = er_gen(n = 50, p = 0.8):
    Erdos-Renyi model: the probability of forming an edge between any two nodes is p = 0.8

2) func = bagen(n = 50, k = 6):
    Barabasi-Albert model: a new node is connected to m = 6 existing nodes.

The output is a 50*50 adjacent matrix, indicating the link status of the nodes in network.

confusion here: i) how to determine the sequence of the nodes generated in adjecent matrix?
ii) how to decide in what order to choose the node to link to in the Barabasi-Albert model?
(we assume picking links one by one)

"""

import numpy as np
import pandas as pd
from numpy import genfromtxt
import networkx as nx
import matplotlib.pyplot as plt
from net_feas import *

"""shortcut to generate random networks is provided by networkx library
source: 
    http://mae.engr.ucdavis.edu/dsouza/Classes/289-S14/NetworkxDemo.pdf
    https://www.cl.cam.ac.uk/~cm542/teaching/2010/stna-pdfs/stna-lecture8.pdf
    https://inst.eecs.berkeley.edu/~ee126/fa14/lab/Lab9_RandomGraphs.pdf
    https://andrewmellor.co.uk/blog/articles/2014/12/14/d3-networks/
"""


def get_test():
    # ajtest = genfromtxt('dat/aj_matrix.xlsx', delimiter=",")
    df = pd.read_csv('dat/aj_matrix.csv', encoding = 'unicode_escape', delimiter=',', header=None)
    ajmtest = df.values
    # print(ajmtest.shape)
    df = pd.read_csv('dat/vector.csv', encoding='unicode_escape', header=None)
    vectest = df.values
    # print(vectest.shape)
    return ajmtest, vectest


def er_gen(n=50, p = 0.5):
    """
    Erdos-Renyi model: the probability of forming an edge between any two nodes is p = 0.5
    :param n: the size of network
    :param p: the probability of the new node connecting to each existing node
    :return: label: 0;  feature vector: [deg_ave, deg_var, clst_ave, clst_var]
    """
    # ajm = np.zeros([n, n])
    g = nx.erdos_renyi_graph(n, p)
    ajm = nx.adjacency_matrix(g, nodelist=range(n))
    aj = ajm.todense()

    # deg = degr_net(ajm)
    # clst = clust_coef(ajm)
    # print(deg)
    # print(clst)

    label = 0
    feature = fea_gen(aj)

    return label, feature


def ba_gen(n = 50, k = 6):
    """
    Barabasi-Albert model: a new node is connected to m = 6 existing nodes.
    :param n: the size of network
    :param k: the number of existing nodes the new node connects to
    :return: label: 1; feature vector: [deg_ave, deg_var, clst_ave, clst_var]
    """
    assert n > k
    g = nx.barabasi_albert_graph(n,k)
    ajm = nx.adjacency_matrix(g, nodelist=range(n))
    aj = ajm.todense()

    # deg = degr_net(ajm)
    # clst = clust_coef(ajm)
    # print(deg)
    # print(clst)

    label = 1
    feature = fea_gen(aj)
    return label, feature


def dat_gen(m = 100):
    """
    to prepare the dataset for training and testing the model
    :param m: the number of samples from each generator [label, feature]
    :return: labels, features
    """
    labels = []
    features = []
    generators = [er_gen, ba_gen]
    for gen in generators:
        for ii in range(m):
            label, feature = gen()
            labels.append(label)
            features.append(feature)
    # print(labels)
    # print(features)
    labels = np.array(labels)
    features = np.array(features)

    return labels, features



def peek_net():
    ajm_ba = ba_gen(50, 6)
    ajm_er = er_gen(50, 0.8)
    return


# peek_net()






"""
for N in [100000, 10000, 1000, 100]:
    g = nx.barabasi_albert_graph(N, 2) #generate a Barabasi-Albert graph with N nodes with eac
    k = np.sort(g.degree().values()) #get the degree of all nodes in the graph and sort them
    y = 1 - np.arange(1., N+1)/N # probability in terms of a rank order
    plt.loglog(k, y, '.', label='N') # plot the CCDF. This makes the tail easy to visualize
    plt.xlabel('k')
    plt.ylabel('CCDF')
    plt.grid() #switch on grid
    plt.legend(loc=0) #show the legend and use the best location for it

"""

