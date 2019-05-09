"""
plan to simulate two types of diffusion models.
    The first model is known as
        independent cascade model
        Class Indcsc
    and
    the second proportional model
        Class Proptn

structure defaultdict list S to indicate whether activated, 0:unactive; 1: active


features to descript influence:
    the active and inactive means of
        1. Degree of each node;
        2. Number of each node’s neighbors that are active;
        3. Proportion of each node’s neighbors that are active;
        4. Number of triangles each node is involved ;
        totally 8 dimensions.
"""

# 1 randomly pick a node among the network to start iteration
#     rand pick a number in an array

# 2 randomly generate value to decide whether infect
# 3 update state, and repeat step 2

import numpy as np
from collections import defaultdict
from net_feas import *
from net_gen import *
import h5py

def act_list(n, graph, state):
    act_list = []
    for ii in range(n):
        vex_ii = str(ii)
        n_act = 0
        nei_list = graph[vex_ii]
        for nei in nei_list:
            if state[nei] == 1:
                n_act += 1
        act_list.append(n_act)
    return act_list


class Diffusion:

    def __init__(self, ajm, nod_0, type, state=None):
        self.ajm = ajm
        self.type = type
        pass

    def prob_update(self, node):
        pass

    def state_update(self):
        """Return the balance remaining after depositing *amount*
        dollars."""
        n = self.n
        for ii in np.arange(n):
            vex_ii = str(ii)
            if self.sta_curr[vex_ii] != 1:
                p = self.prob_update(vex_ii)
                s = np.random.choice([0, 1], p=[1 - p, p])
                self.sta_temp[vex_ii] = s
            else:
                self.sta_temp[vex_ii] = 1

        for ii in np.arange(n):
            vex_ii = str(ii)
            self.sta_prev[vex_ii] = self.sta_curr[vex_ii]
            self.sta_curr[vex_ii] = self.sta_temp[vex_ii]

        # print(self.sta_prev)
        # print(self.sta_curr)
        # self.fea_gen()

        return

    def state_ite(self):
        iteration = self.itera
        for ite in np.arange(iteration):
            self.state_update()

    def fea_gen(self):
        """
        ""
        features to descript influence:
        the active and inactive means of
        1. Degree of each node: deg_ac, deg_ina
        2. Number of each node’s neighbors that are active: nei_ac, nei_ina
        3. Proportion of each node’s neighbors that are active: por_ac, por_ina
        4. Number of triangles each node is involved: tri_ac, tri_ina
        totally 8 dimensions
        :return:

        """

        n = self.n
        deg_list = degr_net(self.ajm)
        graph = self.graph
        state = self.sta_curr
        ac_list = act_list(n, graph, state)
        por_list = [ac / deg for deg, ac in zip(deg_list, ac_list)]
        tri_list = trg_list(self.ajm)
        count_ac = 0
        count_ina = 0
        deg_ac, deg_ina = 0, 0
        nei_ac, nei_ina = 0, 0
        por_ac, por_ina = 0, 0
        tri_ac, tri_ina = 0, 0
        for ii in range(n):
            vex_ii = str(ii)
            if state[vex_ii] == 1:
                count_ac += 1
                deg_ac += deg_list[ii]
                nei_ac += ac_list[ii]
                por_ac += por_list[ii]
                tri_ac += tri_list[ii]

            elif state[vex_ii] == 0:
                count_ina += 1
                deg_ina += deg_list[ii]
                nei_ina += ac_list[ii]
                por_ina += por_list[ii]
                tri_ina += tri_list[ii]

        if count_ac > 0:
            deg_ac = deg_ac / count_ac
            nei_ac = nei_ac / count_ac
            por_ac = por_ac / count_ac
            tri_ac = tri_ac / count_ac
        else:
            state

        if count_ina > 0:
            deg_ina = deg_ina / count_ina
            nei_ina = nei_ina / count_ina
            por_ina = por_ina / count_ina
            tri_ina = tri_ina / count_ina
        else:
            deg_ina, nei_ina, por_ina, tri_ina = 0, 0, 0, 0

        fea = [deg_ac, nei_ac, por_ac, tri_ac, deg_ina, nei_ina, por_ina, tri_ina]

        def fea_join(fea_new, fea0):
            for fe in fea_new:
                fea0.append(fe)
            return fea0

        fea = []
        fea = fea_join(ac_list, fea)
        fea = fea_join(por_list, fea)

        fea = np.array(fea)
        # print(fea)
        return fea, self.type


class Indcsc(Diffusion):
    """A customer of ABC Bank with a checking account. Customers have the
    following properties:

    Attributes:
        name: A string representing the customer's name.
        balance: A float tracking the current balance of the customer's account.
    """

    def __init__(self, ajm, nod_0='0', prob=0.75, itera=20, state=None):
        # print(self)
        self.type = 0
        self.ajm = ajm
        self.n = tot_num(ajm)
        self.graph = aj2graph(ajm)
        self.prob = prob
        self.itera = itera
        self.sta_curr = defaultdict(list)
        self.sta_prev = defaultdict(list)
        self.sta_temp = defaultdict(list)
        for ii in range(self.n):
            self.sta_curr[str(ii)] = 0
            self.sta_prev[str(ii)] = 0
        self.sta_curr[nod_0] = 1
        if state is not None:
            for ii in range(self.n):
                v_ii = str(ii)
                self.sta_curr[v_ii] = state[ii]

    def prob_update(self, node):

        neib_list = self.graph[node]
        n_active = 0
        for neib in neib_list:
            act = self.sta_curr[neib] - self.sta_prev[neib]
            n_active += act
        p = self.prob
        proba = 1 - (1 - p) ** n_active
        return proba


    def state_update(self):
        """Return the balance remaining after depositing *amount*
        dollars."""
        n = self.n
        for ii in np.arange(n):
            vex_ii = str(ii)
            if self.sta_curr[vex_ii] != 1:
                p = self.prob_update(vex_ii)
                s = np.random.choice([0,1], p=[1-p, p])
                self.sta_temp[vex_ii] = s
            else:
                self.sta_temp[vex_ii] = 1

        for ii in np.arange(n):
            vex_ii = str(ii)
            self.sta_prev[vex_ii] = self.sta_curr[vex_ii]
            self.sta_curr[vex_ii] = self.sta_temp[vex_ii]

        # print(self.sta_prev)
        # print(self.sta_curr)
        # self.fea_gen()
        return

    def state_ite(self):
        iteration = self.itera
        for ite in np.arange(iteration):
            self.state_update()

    def fea_gen(self):
        """
        ""
        features to descript influence:
        the active and inactive means of
        1. Degree of each node: deg_ac, deg_ina
        2. Number of each node’s neighbors that are active: nei_ac, nei_ina
        3. Proportion of each node’s neighbors that are active: por_ac, por_ina
        4. Number of triangles each node is involved: tri_ac, tri_ina
        totally 8 dimensions
        :return:
        """
        n = self.n
        deg_list = degr_net(self.ajm)
        graph = self.graph
        state = self.sta_curr
        ac_list = act_list(n, graph, state)
        por_list = [ac / deg for deg, ac in zip(deg_list, ac_list)]
        tri_list = trg_list(self.ajm)
        count_ac = 0
        count_ina = 0
        deg_ac, deg_ina = 0, 0
        nei_ac, nei_ina = 0, 0
        por_ac, por_ina = 0, 0
        tri_ac, tri_ina = 0, 0
        for ii in range(n):
            vex_ii = str(ii)
            if state[vex_ii] == 1:
                count_ac += 1
                deg_ac += deg_list[ii]
                nei_ac += ac_list[ii]
                por_ac += por_list[ii]
                tri_ac += tri_list[ii]

            elif state[vex_ii] == 0:
                count_ina += 1
                deg_ina += deg_list[ii]
                nei_ina += ac_list[ii]
                por_ina += por_list[ii]
                tri_ina += tri_list[ii]

        if count_ac > 0:
            deg_ac = deg_ac / count_ac
            nei_ac = nei_ac / count_ac
            por_ac = por_ac / count_ac
            tri_ac = tri_ac / count_ac
        else:
            state

        if count_ina > 0:
            deg_ina = deg_ina / count_ina
            nei_ina = nei_ina / count_ina
            por_ina = por_ina / count_ina
            tri_ina = tri_ina / count_ina
        else:
            deg_ina = 0
            nei_ina = 0
            por_ina = 0
            tri_ina = 0

        fea = [deg_ac, nei_ac, por_ac, tri_ac, deg_ina, nei_ina, por_ina, tri_ina]

        def fea_join(fea_new, fea0):
            for fe in fea_new:
                fea0.append(fe)
            return fea0

        fea = fea_join(ac_list, fea)


        fea = np.array(fea)
        print(fea.shape)
        return fea, self.type





class Proptn(Diffusion):
    """A customer of ABC Bank with a checking account. Customers have the
    following properties:

    Attributes:
        name: A string representing the customer's name.
        balance: A float tracking the current balance of the customer's account.
    """

    def __init__(self, ajm, nod_0='0', alpha=0.3, itera=20, state=None):
        # print(self)
        self.type = 1
        self.ajm = ajm
        self.n = tot_num(ajm)
        self.graph = aj2graph(ajm)
        self.alpha = alpha
        self.itera = itera
        self.sta_curr = defaultdict(list)
        self.sta_prev = defaultdict(list)
        self.sta_temp = defaultdict(list)
        for ii in range(self.n):
            self.sta_curr[str(ii)] = 0
            self.sta_prev[str(ii)] = 0
        self.sta_curr[nod_0] = 1
        if state is not None:
            for ii in range(self.n):
                v_ii = str(ii)
                self.sta_curr[v_ii] = state[ii]

    def prob_update(self, node):

        neib_list = self.graph[node]
        d = len(neib_list)

        n_active = 0
        for neib in neib_list:
            act = self.sta_curr[neib]
            n_active += act
        alpha = self.alpha
        proba = alpha * n_active / d
        return proba


    def state_update(self):
        """Return the balance remaining after depositing *amount*
        dollars."""
        n = self.n
        for ii in np.arange(n):
            vex_ii = str(ii)
            if self.sta_curr[vex_ii] != 1:
                p = self.prob_update(vex_ii)
                s = np.random.choice([0,1], p=[1-p, p])
                self.sta_temp[vex_ii] = s
            else:
                self.sta_temp[vex_ii] = 1

        for ii in np.arange(n):
            vex_ii = str(ii)
            self.sta_prev[vex_ii] = self.sta_curr[vex_ii]
            self.sta_curr[vex_ii] = self.sta_temp[vex_ii]

        # print(self.sta_prev)
        # print(self.sta_curr)
        # self.fea_gen()

        return

    def state_ite(self):
        iteration = self.itera
        for ite in np.arange(iteration):
            self.state_update()


    def fea_gen(self):
        """
        ""
        features to descript influence:
        the active and inactive means of
        1. Degree of each node: deg_ac, deg_ina
        2. Number of each node’s neighbors that are active: nei_ac, nei_ina
        3. Proportion of each node’s neighbors that are active: por_ac, por_ina
        4. Number of triangles each node is involved: tri_ac, tri_ina
        totally 8 dimensions
        :return:

        """


        n = self.n
        deg_list = degr_net(self.ajm)
        graph = self.graph
        state = self.sta_curr
        state_top = [state[str(nod)] for nod in range(n)]
        ac_list = act_list(n, graph, state)
        por_list = [ac/deg for deg, ac in zip(deg_list, ac_list)]
        tri_list = trg_list(self.ajm)
        count_ac = 0
        count_ina = 0
        deg_ac, deg_ina = 0, 0
        nei_ac, nei_ina = 0, 0
        por_ac, por_ina = 0, 0
        tri_ac, tri_ina = 0, 0
        for ii in range(n):
            vex_ii = str(ii)
            if state[vex_ii] == 1:
                count_ac += 1
                deg_ac += deg_list[ii]
                nei_ac += ac_list[ii]
                por_ac += por_list[ii]
                tri_ac += tri_list[ii]

            elif state[vex_ii] == 0:
                count_ina += 1
                deg_ina += deg_list[ii]
                nei_ina += ac_list[ii]
                por_ina += por_list[ii]
                tri_ina += tri_list[ii]


        if count_ac > 0:
            deg_ac = deg_ac / count_ac
            nei_ac = nei_ac / count_ac
            por_ac = por_ac / count_ac
            tri_ac = tri_ac / count_ac
        else:
            state

        if count_ina > 0 :
            deg_ina = deg_ina / count_ina
            nei_ina = nei_ina / count_ina
            por_ina = por_ina / count_ina
            tri_ina = tri_ina / count_ina
        else:
            deg_ina, nei_ina, por_ina, tri_ina = 0, 0, 0, 0

        fea = [deg_ac, nei_ac, por_ac, tri_ac, deg_ina, nei_ina, por_ina, tri_ina]

        def fea_join(fea_new, fea0):
            for fe in fea_new:
                fea0.append(fe)
            return fea0
        fea = []
        fea = fea_join(state_top, fea)
        fea = fea_join(ac_list, fea)
        # fea = fea_join(por_list, fea)


        fea = np.array(fea)
        # print(fea)
        return fea, self.type



def diffu_fea_gen(n_samples=200):
    aj_test, sta_test = get_test()
    n = tot_num(aj_test)
    labels = []
    features = []
    for ii in range(n_samples):
        ix = ii % n
        v_ii = str(ix)
        tt = Indcsc(aj_test, v_ii, prob=0.75)
        tt.state_ite()
        fea_ind, lbl0 = tt.fea_gen()
        features.append(fea_ind)
        labels.append(lbl0)

        rr = Proptn(aj_test, v_ii, alpha=0.3)
        rr.state_ite()
        fea_pro, lbl1 = rr.fea_gen()
        features.append(fea_pro)
        labels.append(lbl1)
        del tt
        del rr

    features = np.array(features)
    labels = np.array(labels)

    # hf = h5py.File('dat/diffu_fea.dat', 'w')
    # hf.create_dataset('features', data=features)
    # hf.create_dataset('labels', data=labels)

    return features, labels


# diffu_fea_gen()

# aj_test, sta_test = get_test()
#
#
# deg_test = degr_net(aj_test)
# print(deg_test)
# clst_test = clust_coef(aj_test)
# print(clst_test)
# print(np.sum(sta_test))
# print(aj_test)
# print(aj2graph(aj_test))
# print(degr_net(aj_test))

