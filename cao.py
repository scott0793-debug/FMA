import gurobipy as gp
from gurobipy import GRB
import numpy as np
import func
import time
from evaluation import calc_path_prob
from fma import calc_path_g

def PLM(mymap, S, T, phi=10, e=0.1):
    g_best = -10**7
    g_best_last = -10**7
    probability_last = 0
    max_path = 0
    lmd = np.random.random([S, 1])
    
    samples = func.generate_samples(mymap, S)
    
    T = np.ones([S, 1]) * T

    k = 1
    k_x = 0

    while(True):
        d_cost, path, x = func.dijkstra(mymap.G, mymap.r_0, mymap.r_s, ext_weight=np.dot(samples, lmd))
        sub1_cost = d_cost - np.dot(T.T, lmd)

        tmp = np.ones([S, 1])-lmd
        xi = np.where(tmp > 0, 0, 10**7)
        sub2_cost = np.sum(np.dot(tmp.T, xi))

        cost = sub1_cost + sub2_cost
        probability = np.sum(np.where(np.dot(samples.T, x) <= T, 1, 0)) / samples.shape[1]

        if probability >= probability_last:
            max_path = path
            probability_last = probability
        probability = max(probability, probability_last)
        # print(k)
        # print(cost)
        # print(max_path)

        g_best = max(cost, g_best)
        if (g_best - g_best_last >= e):
            k_x = k
        g_best_last = g_best

        if(k-k_x >= phi):
            break

        d_g = np.dot(samples.T, x) - T - xi

        alpha = 0.0001/np.sqrt(k)
        lmd += alpha * d_g
        lmd = np.where(lmd > 0, lmd, 0)
        
        k += 1

    # print("final path:" + str(np.array(max_path) + 1))
    return probability, max_path

def ILP(mymap, S, T):
    V = 10**4

    samples = func.generate_samples(mymap, S).T

    obj_temp1 = np.zeros(mymap.n_link)
    obj_temp2 = np.ones(S)
    obj = np.hstack((obj_temp1, obj_temp2))

    eq_temp = np.zeros([mymap.n_node, S])
    eq_constr = np.hstack((mymap.M, eq_temp))

    ineq_temp = -V * np.eye(S)
    ineq_constr = np.hstack((samples, ineq_temp))

    T = np.ones(S) * T

    n_elem = mymap.n_link + S

    m = gp.Model("ilp")
    m.Params.LogToConsole = 0

    z = m.addMVar(shape=n_elem, vtype=GRB.BINARY, name="z")
    m.setObjective(obj @ z, GRB.MINIMIZE)
    m.addConstr(ineq_constr @ z <= T, name="ineq")
    m.addConstr(eq_constr @ z == mymap.b.reshape(-1), name="eq")
    m.optimize()

    res = z.X
    # print(res)

    prob = 1 - np.dot(obj.T, res).item()/S
    path = np.flatnonzero(res[:mymap.n_link])
    path = func.sort_path_order(path, mymap)
    # print(prob)

    # print("final path:" + str(path + 1))
    return prob, path

def other_iterations(alg, mymap, T, S, MaxIter):
    '''
    run a certain algorithm 'alg' for MaxIter times and return the statistics
    '''

    pro = []
    g = []
    t_delta = []
    g_delta = []
    for ite in range(MaxIter):
        print('{} iteration #{}'.format(alg.__name__, ite))
        t1 = time.perf_counter()
        _, path = alg(mymap, S, T)
        t_delta.append(time.perf_counter() - t1)
        print("final path: {}\n".format(str(np.array(path) + 1)))
        pro.append(calc_path_prob(path, mymap, T))
        g.append(calc_path_g(path, mymap, T))
        g_delta.append(1-calc_path_g(path, mymap, T))
    return np.mean(pro), np.mean(g), np.mean(t_delta), np.max(t_delta), np.mean(g_delta)
