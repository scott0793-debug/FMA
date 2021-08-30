import numpy as np
import func
import time
import os
from copy import deepcopy
from cao import ILP, PLM, other_iterations
from prakash import DOT
from fma import FMA_MAP, FMA, LET_path

'''
Note: The indexes of nodes and links start from 1 when being set or displayed, but start from 0 when stored and calculated.
'''

np.random.seed(808)
def record(method, res, file_name):
    fp = open(file_name, 'a+')
    print(method + " prob, g, t, t_max, g_delta: " + str(res) + "\n")
    print(time.asctime())
    fp.write(method + " prob, g, t, t_max, g_delta: " + str(res) + "\n")
    fp.close()

def write_file(content_name, content, file_name):
    fp = open(file_name, 'a+')
    fp.write("{}={}\n".format(content_name,content))
    fp.close()


curr_model = "G"
K = 100 #FMA
S = 100
MaxIter = 100
DOT_delta = 0.005#0.005 #0.01 0.05 2
PA_maxspeed = 3 #2

test_name = curr_model + " XXX"
curr_dir = os.getcwd()
map_dir = curr_dir + '/Networks/'
map_id = 1 #map_id can be integers from 0~4

file_name = test_name+'.txt' #results are stored in this file
file_list = [file_name]
fp = open(file_name, 'a+')
fp.write("model={}\n".format(test_name))
fp.write("K={}\n".format(K))
fp.write("S={}\n".format(S))
fp.write("MaxIter={}\n".format(MaxIter))
fp.write("DOT_delta={}\n".format(DOT_delta))
fp.write("PA_maxspeed={}\n".format(PA_maxspeed))
fp.write("\n")
fp.close()

T_factors = [1, 1.05, 1.1]
kappas = [0.15, 0.25, 0.5]

OD_pairs = [[1, 6]]

for kappa in kappas:
    mymap = FMA_MAP()
    mymap.generate_real_map(map_id, map_dir, kappa)
    write_file("============", "=============", file_name)
    write_file("kappa", kappa, file_name)
    write_file("============", "=============", file_name)
    sub_results = {"FMA":[], "DOT":[], "PA":[], "ILP":[], "PLM":[]}
    results = {}
    p_mean = {}
    g_mean = {}
    t_mean = {}

    for tf in T_factors:
        results[tf] = deepcopy(sub_results)
        p_mean[tf] ={}
        g_mean[tf] ={}
        t_mean[tf] ={}

    for OD in OD_pairs:
        write_file("OD", OD, file_name)
        write_file("============", "=============", file_name)
        print("OD={}".format(OD))
        mymap.update_OD(OD)

        for tf in T_factors:
            T = tf * mymap.dij_cost
        
            write_file("T", T, file_name)
            write_file("=========", "==========", file_name)
            print("========================================================================================")
            print("T={}".format(T))
            print("tf={}".format(tf))

        # ##############################-----------FMA-----------####################################################
            FMA_Solver = FMA(mymap, T, K)
            prob, g, t_delta = FMA_Solver.policy_iteration()[:3]
            g_delta = 1 - g
            res_FMA = prob, g, t_delta, 0, g_delta
            results[tf]["FMA"].append(res_FMA)
            record("FMA", res_FMA, file_name)
        # # ##############################-----------DOT-----------###################################################
            DOT_Solver = DOT(mymap, T, DOT_delta)
            path, cost, g, prob=DOT_Solver.policy2path(T)
            g_delta = 1 - g
            res_DOT = prob, g, DOT_Solver.DOT_t_delta, 0, g_delta
            results[tf]["DOT"].append(res_DOT)
            record("DOT", res_DOT, file_name)  
        # # ##############################-----------PA-----------####################################################
            prob, g, t_delta = DOT_Solver.PA(T, PA_maxspeed)
            g_delta = 1 - g
            res_PA = prob, g, t_delta, 0, g_delta 
            results[tf]["PA"].append(res_PA)
            record("PA", res_PA, file_name)
        # # #############################-----------ILP-----------###########################################
            res_ILP = other_iterations(ILP, mymap, T, S, MaxIter) 
            results[tf]["ILP"].append(res_ILP)
            record("ILP", res_ILP, file_name)
        # ##############################-----------PLM-----------####################################################
            res_PLM = other_iterations(PLM, mymap, T, S, MaxIter)
            results[tf]["PLM"].append(res_PLM)
            record("PLM", res_PLM, file_name)
        # # ##############################-----------LET-----------####################################################
            res_LET = LET_path(mymap, T)
            results[tf]["LET"].append(res_LET)
            record("LET", res_LET, file_name)

            fp = open(file_name, 'a+')
            fp.write("\n\n")
            fp.close()

    fp = open(file_name, 'a+')
    for tf in results.keys():
        fp.write("tf={}\n".format(tf))
        for alg in results[tf].keys():
            ret = results[tf][alg]
            fp.write(alg+"_results={}\n".format(ret))
            p_mean[tf][alg] = np.mean(np.array(ret)[:,0])
            g_mean[tf][alg] = np.mean(np.array(ret)[:,1])
            t_mean[tf][alg] = np.mean(np.array(ret)[:,2])
        fp.write("\n\n")

    fp.write("p_mean={}\n".format(p_mean))
    fp.write("g_mean={}\n".format(g_mean))
    fp.write("t_mean={}\n".format(t_mean))

    fp.close()
