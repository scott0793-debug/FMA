from fma import calc_path_g
import numpy as np
import pandas as pd
import func
import time
from scipy.stats import norm
from evaluation import calc_path_prob

class DOT:
    def __init__(self, mymap, T, delta):
        self.map = mymap
        self.T = T
        self.delta = delta
        self.model = mymap.model

        t1 = time.perf_counter()
        self.DOT_Policy()
        self.DOT_t_delta = time.perf_counter() - t1

    def DOT_Policy(self):
        n_timestamp = np.ceil(self.T/self.delta).astype(int)
        self.delta = self.T / n_timestamp
        J = np.zeros([self.map.n_node, n_timestamp+1])
        J[self.map.r_s,:] = 1
        U = -1 * np.ones([self.map.n_node, n_timestamp+1])
        times = np.linspace(0, self.T, num=n_timestamp+1).reshape(-1,1)
        # times = np.arange(0, T, delta).reshape(-1,1)

        if self.model == 'G':
            CDF = norm.cdf(times, self.map.mu.reshape(-1), np.sqrt(np.diag(self.map.cov)))
            self.CDF_delta = CDF[1:, :] - CDF[:n_timestamp, :]

        for timestamp in range(n_timestamp-1,-1,-1):
            # if timestamp%1000 == 0:
            #     print(timestamp)
            for node in self.map.G.nodes:
                if node != self.map.r_s:
                    prob_max = 0
                    u = -1
                    for _, next_node, d in self.map.G.out_edges(node, data=True):
                        link_idx = d['index']
                        prob = np.dot(self.CDF_delta[:n_timestamp-timestamp, link_idx], J[next_node, timestamp+1:n_timestamp+1])
                        if prob >= prob_max:
                            prob_max = prob
                            u = link_idx
                    
                    J[node, timestamp] = prob_max
                    U[node, timestamp] = u

        self.J = J
        self.U = U.astype(int)
        

    def get_DOT_prob(self, t=None):
        if t is None:
            t = self.T
        stamp = self._t2stamp(t)
        return self.J[self.map.r_0, stamp].item()

    def PA(self, t, maxspeed):
        stamp = self._t2stamp(t)
        J = self.J[:,stamp:]
        self.maxspeed = maxspeed

        t1 = time.perf_counter()
        PA_prob, path = self._PA(t, J)
        PA_t_delta = time.perf_counter() - t1

        PA_g = calc_path_g(path, self.map, t)

        return PA_prob, PA_g, PA_t_delta

    def _PA(self, t, J):
        lb = calc_path_prob(self.map.dij_path, self.map, t)
        lb *= 0.8

        empty_df = pd.DataFrame(columns=['pre_node', 'pre_sub', 'link_idx', 'min_time'])
        df = pd.DataFrame({"pre_node":None, "pre_sub":None, "link_idx":None, "min_time":0}, index=[0])
        self.PI = {self.map.r_0:df}
        L = [self.map.r_0]

        while L:
            pre_node = L.pop(0)

            for _, next_node, d in self.map.G.out_edges(pre_node, data=True):
                is_empty = 0
                if next_node not in self.PI:
                    self.PI[next_node] = empty_df
                if self.PI[next_node].empty:
                    is_empty = 1
                link_idx = d['index']
                modified_flag = 0

                for pre_sub, row in self.PI[pre_node].iterrows():
                    if not is_empty:
                        if self._is_cyclic(next_node, pre_node, pre_sub):
                            continue
                        if self._is_explored(next_node, pre_node, pre_sub, link_idx):
                            continue

                    df = self._add_subpath(pre_node, pre_sub, link_idx, row['min_time'])
                    if df["min_time"] >= J.shape[1]:
                        continue

                    ub1 = J[next_node, df["min_time"]]
                    if ub1 < lb:
                        continue
                    ub2 = self._calc_ub2(next_node, pre_node, pre_sub, link_idx, J)
                    ub2 *= 1.2
                    if lb > ub2:
                        continue

                    self.PI[next_node] = self.PI[next_node].append(df, ignore_index=True)
                    modified_flag = 1

                if modified_flag and next_node not in L:
                    L.append(next_node)

        #Extract candidate path
        paths = []
        probs = []

        if self.map.r_s not in self.PI or self.PI[self.map.r_s].empty:
            print("warning!!!!!!")
            for path_node in func.k_shortest_paths(self.map, k=5):
                path = []
                for j in range(len(path_node)-1):
                    path.append(self.map.G[path_node[j]][path_node[j+1]]['index'])

                prob = calc_path_prob(path, self.map, t)
                paths.append(path)
                probs.append(prob)
                # print(path)
                # print(prob)
        else:
            for _, temp in self.PI[self.map.r_s].iterrows():
                pre_node = temp["pre_node"]
                pre_sub = temp["pre_sub"]
                path = []

                while pre_node is not None:
                    path.append(temp['link_idx'])
                    temp = self.PI[pre_node].loc[pre_sub]
                    pre_node = temp['pre_node']
                    pre_sub = temp['pre_sub']
                
                path.reverse()
                prob = calc_path_prob(path, self.map, t)
                paths.append(path)
                probs.append(prob)
                # print(path)
                # print(prob)

        #Find the Path with MPOA
        MPOA = np.max(probs)
        MPOA_path = paths[np.argmax(probs)]
        print("Final")
        print(MPOA_path)
        print(MPOA)

        return MPOA, MPOA_path

    def _is_cyclic(self, next_node, pre_node, pre_sub):
        while pre_node is not None:
            if next_node == pre_node:
                return True
            temp = self.PI[pre_node].loc[pre_sub]
            pre_node = temp['pre_node']
            pre_sub = temp['pre_sub']
        return False
        
    def _is_explored(self, next_node, pre_node, pre_sub, link_idx):
        df = self.PI[next_node]
        return not df[(df["pre_node"] == pre_node) & (df["pre_sub"] == pre_sub) & (df["link_idx"] == link_idx)].empty

    def _calc_ub2(self, next_node, pre_node, pre_sub, link_idx, J):
        H = J[next_node, :].reshape(-1)
        # size = H.size-1
        # mask = np.flip(np.triu(np.ones((size, size)), k=0), axis=1)
        while pre_node is not None:
            H_new = np.zeros_like(H)
            for i in range(H.size-1):
                H_new[i] = np.dot(self.CDF_delta[:H.size-1-i, link_idx], H[i+1:])
                if H_new[i] < 1e-4:
                    break
            H = H_new

            temp = self.PI[pre_node].loc[pre_sub]
            pre_node = temp['pre_node']
            pre_sub = temp['pre_sub']
            link_idx = temp["link_idx"]

        return H[0]

    def _add_subpath(self, pre_node, pre_sub, link_idx, pre_min_time):
        min_time = pre_min_time + np.floor(self.map.mu[link_idx].item()/self.maxspeed/self.delta).astype(int)
        return pd.Series({"pre_node":pre_node, "pre_sub":pre_sub, "link_idx":link_idx, "min_time":min_time})

    def _t2stamp(self, t):
        return round((self.T - t)/self.delta)

    def policy2path(self,t): 
        stamp = self._t2stamp(t)
        U = self.U[:,stamp:]

        path = []
        cost = 0
        g = 0
        cost_stamp=0
        node =self.map.r_0
        node_list = [node]
        print('max stamp',U.shape[1]-1)
        while cost_stamp<U.shape[1]-1:
            
            cost += self.map.mu[U[node,cost_stamp]]
            path.append(U[node,cost_stamp])
            next_node = func.find_next_node(self.map, node, U[node,cost_stamp])
            node_list.append(next_node)
            node = next_node
            cost_stamp = np.ceil(cost/self.delta)
            cost_stamp = cost_stamp[0].astype(int)
            print('curr stamp', cost_stamp)
            if node == self.map.r_s:
                break
        print('nodelist',[x+1 for x in node_list])

        if node != self.map.r_s:
            d_cost, d_path, _ = func.dijkstra(self.map.G, node, self.map.r_s)
            path += d_path
            cost += d_cost
        g = calc_path_g(path, self.map, t)
        prob = calc_path_prob(path, self.map, t)
           
        return path, cost, g, prob

        
        
