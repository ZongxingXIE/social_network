"""
In this script, we provide several functions to extract features which is descriptive of network topology:
1. Number of nodes. (this is actually predefined???)
        The total number of nodes in the network.
2. Average degree.
        The average degree of a network is the average of all
        nodes' degrees over the entire network.
3. Degree distribution.
        The degree distribution of a network is the probability
        distribution of these degrees.
4. Clustering coeffcient, including
        a)The local clustering coefficient;
        b)The global clustering coeffcient, which is the average of local clustering coeffcients
        over the network.
5. Diameter.
        The diameter of a graph is the maximum distance between
        any two nodes of the network.

Input: adjacent matrix
Output: corresponding features

"""

import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.special import comb

# 1 return rank of aj matrix
def tot_num(ajm):
    return np.shape(ajm)[0]


# 2 return the degree of the current node
def degr_nod(nod_vec):
    return np.sum(nod_vec)


# 2 return the list of node degree
def degr_net(ajm):

    n = tot_num(ajm)
    degr_list = np.zeros(n)
    for ii in range(n):
        nod_i = ajm[ii, :]
        degr_list[ii] = degr_nod(nod_i)

    return degr_list


def prefer_prob(ajm):

    degr_list = degr_net(ajm)
    degr_tot = degr_list.sum()
    degr_tot = 1 if degr_tot == 0 else degr_tot
    preferential = degr_list / degr_tot

    return preferential


# # 2 return the average degree
# def degr_ave(ajm):
#     degr_list = degr_net(ajm)
#     return np.mean(degr_list)
""" for the average degree, refer to 3 stats of degree distribution"""

# 3 return the distribute of degree
def degr_dstrb(ajm):
    degr_list = degr_net(ajm)
    dmin = np.min(degr_list)
    dmax = np.max(degr_list)
    rng = np.arange(dmin, dmax + 1)
    n = len(rng)
    counter = np.zeros(n)
    for idx, dgr_cs in enumerate(rng):
        for deg in degr_list:
            if dgr_cs == deg:
                counter[idx] += 1

    tot_degr = counter.sum()
    tot_degr = 1 if tot_degr == 0 else tot_degr
    prob_degr = counter / tot_degr

    return prob_degr, rng


# 3 return stats of degree distribution
def degr_stat(ajm):
    # degr_list = degr_net(ajm)
    deg_prob, deg_rng = degr_dstrb(ajm)
    deg_ave = np.dot(deg_prob, deg_rng)
    dev = (deg_rng - deg_ave) ** 2
    deg_var = np.dot(deg_prob, dev)

    return deg_ave, deg_var

# 4 cluster coefficient

def binomial(n, k = 2):
    """
    source: https://stackoverflow.com/questions/3025162/statistics-combinations-in-python/3027128
    Using dynamic programming,
    the time complexity is Θ(n*m) and space complexity Θ(m)
    (int, int) -> int

             | c(n-1, k-1) + c(n-1, k), if 0 < k < n
    c(n,k) = | 1                      , if n = k
             | 1                      , if k = 0

    Precondition: n > k

    >> binomial(9, 2)
    36

    intuition:

                                                   1
                                                1     1
                                             1     2     1
                                          1     3     3     1
                                       1     4     6     4     1
                                    1     5    10    10     5     1
                                 1     6    15    20    15     6     1
                              1     7    21    35    35    21     7     1
                           1     8    28    56    70    56    28     8     1
                        1     9    36    84   126   126    84    36     9     1
                     1    10    45   120   210   252   210   120    45    10     1
                  1    11    55   165   330   462   462   330   165    55    11     1
               1    12    66   220   495   792   924   792   495   220    66    12     1
            1    13    78   286   715  1287  1716  1716  1287   715   286    78    13     1
         1    14    91   364  1001  2002  3003  3432  3003  2002  1001   364    91    14     1
      1    15   105   455  1365  3003  5005  6435  6435  5005  3003  1365   455   105    15     1
    1    16   120   560  1820  4368  8008 11440 12870 11440  8008  4368  1820   560   120    16     1


    """
    if n < k:
        return 1

    c = [0] * (n + 1)
    c[0] = 1
    for i in range(1, n + 1):
        c[i] = 1
        j = i - 1
        while j > 0:
            c[j] += c[j - 1]
            j -= 1

    return c[k]


#
def n_trg(ajm, vex):
    graph = aj2graph(ajm)
    neighbors = graph[vex]
    n_n = len(neighbors)
    count_trg = 0
    for jj in range(n_n - 1):
        potientials = neighbors[jj + 1:]
        vex_jj = neighbors[jj]
        neigb_jj = graph[vex_jj]
        for vex_n in neigb_jj:
            if vex_n in potientials:
                count_trg += 1

    return count_trg



def trg_list(ajm):
    n = tot_num(ajm)
    tri_list = [0] * n
    graph = aj2graph(ajm)
    for ii in range(n):
        vex_ii = str(ii)
        linkedin = graph[vex_ii]
        degr = len(linkedin)

        gamma = n_trg(ajm, vex_ii)

        tri_list[ii] = gamma

    return tri_list


# return cluster coefficient list (triangle portion)
def clust_coef(ajm):
    n = tot_num(ajm)
    coef_list = [0] * n
    graph = aj2graph(ajm)
    for ii in range(n):
        vex_ii = str(ii)
        linkedin = graph[vex_ii]
        degr = len(linkedin)
        tau = binomial(degr, 2)
        # print(tau)
        gamma = n_trg(ajm, vex_ii)
        tau = 1 if tau == 0 else tau
        coef_list[ii] = gamma / tau

    return coef_list





# return global cluster coefficient C
def clust_g_coef(ajm):
    coef_list = clust_coef(ajm)
    clst_ave = np.mean(coef_list)
    clst_var = np.var(coef_list)
    return clst_ave, clst_var


# 5 from aj matrix to graph
def aj2graph(ajm):
    """
    source: https://stackoverflow.com/questions/29320556/finding-longest-path-in-a-graph
    """
    graph = defaultdict(list)
    n = tot_num(ajm)
    for ii in range(n):
        for jj in range(ii, n):
            if ajm[ii, jj] == 1:
                vex_ii = str(ii)
                vex_jj = str(jj)
                graph[vex_ii].append(vex_jj)
                graph[vex_jj].append(vex_ii)
    return graph


# 5 return depth-first search (DFS) path
def dfs_path(G,v,seen=None,path=None):
    """
    source: https://stackoverflow.com/questions/29320556/finding-longest-path-in-a-graph
    """
    if seen is None: seen = []
    if path is None: path = [v]

    seen.append(v)

    paths = []
    for t in G[v]:
        if t not in seen:
            t_path = path + [t]
            paths.append(tuple(t_path))
            paths.extend(dfs_path(G, t, seen[:], t_path))

    return paths


# 5 return the length of longest path (diameter)
def diam_aj(ajm):
    maxima = 0
    n = tot_num(ajm)
    graph = aj2graph(ajm)
    for idx in range(n):
        vex = str(idx)
        paths = dfs_path(graph, vex)
        lenp_list = [len(p) for p in paths]
        if not lenp_list:
            max_len = 0
        else:
            max_len = max(lenp_list)

        if maxima < int(max_len):
            maxima = max_len

    return maxima


# Feature generator
def fea_gen(ajm):
    """
    :param ajm:
    :return:
    1. n_n (int):
        Number of nodes. (this is actually predefined???)
        The total number of nodes in the network.
    2. deg_ave (float):
        Average degree.
        The average degree of a network is the average of all
        nodes' degrees over the entire network.
    3. deg_var (float):
        Degree distribution.
        The degree distribution of a network is the probability
        distribution of these degrees.
    4. clst_g_coef (float):
        Clustering coeffcient, including
        a)The local clustering coefficient;
        b)The global clustering coeffcient, which is the average of local clustering coeffcients
        over the network.
    5. diam (int):
        Diameter.
        The diameter of a graph is the maximum distance between
        any two nodes of the network.

        however, as I notice, the n_n and diam are always the same given the same size of aj matrix.
        so it is not informative to distinguish generator models, thus not using them as features.
    """
    deg_list = degr_net(ajm)
    deg_p5 = np.percentile(deg_list,5)
    deg_p25 = np.percentile(deg_list, 25)
    deg_p50 = np.percentile(deg_list, 50)
    deg_p75 = np.percentile(deg_list, 75)
    deg_p95 = np.percentile(deg_list, 95)
    clst_list = clust_coef(ajm)
    clst_p5 = np.percentile(clst_list, 5)
    clst_p25 = np.percentile(clst_list, 25)
    clst_p50 = np.percentile(clst_list, 50)
    clst_p75 = np.percentile(clst_list, 75)
    clst_p95 = np.percentile(clst_list, 95)

    deg_fea = [deg_p5, deg_p25, deg_p50, deg_p75, deg_p95]
    clst_fea = [clst_p5, clst_p25, clst_p50, clst_p75, clst_p95]

    # n_n = tot_num(ajm)  # not informative
    deg_ave, deg_var = degr_stat(ajm)
    clst_ave, clst_var = clust_g_coef(ajm)
    # diam = diam_aj(ajm) # not informative

    # features = [n_n, deg_ave, deg_var, clst_g_coef, diam]

    features = [deg_ave, deg_var, clst_ave, clst_var]
    # features += deg_fea
    # features += clst_fea
    features = np.array(features)
    return features


def testbench():

    ajm = np.zeros([4,4])
    ajm[1,2] = 1
    ajm[2,1] = 1
    ajm[0,1] = 1
    ajm[1,0] = 1
    ajm[0,2] = 1
    ajm[2,0] = 1
    ajm[1,3] = 1
    ajm[3,1] = 1
    ajm[3,2] = 1
    ajm[2,3] = 1
    print(ajm)

    graph = aj2graph(ajm)
    print(graph)
    #
    path = dfs_path(graph, "1")
    print(path)
    # dia = diam_aj(ajm)
    #
    # print(dia)

    # print(binomial(6,2))
    # print(n_trg(ajm, "1"))
    # print( clust_coef(ajm))
    # print(degr_ave(ajm))
    # print(degr_dstrb(ajm))
    # print(degr_stat(ajm))
    # feature = fea_gen(ajm)
    # print(feature)
    # prefer = prefer_prob(ajm)
    # print(prefer)


# testbench()

