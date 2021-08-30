from networkx.algorithms.graphical import is_multigraphical
import numpy as np
import pandas as pd
from scipy import stats
from heapq import heapify, heappush, heappop
import networkx as nx
from itertools import islice

class Map():
    def __init__(self, model=None):
        self.model = model

    def make_map_with_M(self, mu, cov, M, is_multi=True):
        self.mu = mu
        self.cov = cov
        self.M = M
        self.n_node = self.M.shape[0]
        self.n_link = self.M.shape[1]
        self.G = convert_map2graph(self, is_multi)

    def make_map_with_G(self, mu, cov, G, OD_true):
        self.mu = mu
        self.cov = cov
        self.r_0, self.r_s = OD_true[0], OD_true[1]
        self.G = G
        self.M = None
        self.b = None

    def update_OD(self, OD_ori):
        self.b, self.r_0, self.r_s = generate_b(self.n_node, OD_ori[0], OD_ori[1])
        self.dij_cost, self.dij_path, self.dij_onehot_path = dijkstra(self.G, self.r_0, self.r_s)

    def generate_real_map(self, map_id, map_dir):
        ''' map_id is an integer that identifies the map you wish to use.
            map_dir is the directory you store the networks, which can be download from the link provided in README.md.
            map_id | network
                0    Simple
                1    Sioux Falls
                2    Anaheim
                3    Winnipeg
                4    Chicago-Sketch
        '''
        M, mu, cov = extract_map(map_id, map_dir)
        self.make_map_with_M(mu, cov, M, is_multi=False)
        self.model = "G"
        self.decom = "cholesky"

class priority_dict(dict):
    def __init__(self, *args, **kwargs):
        super(priority_dict, self).__init__(*args, **kwargs)
        self._rebuild_heap()

    def _rebuild_heap(self):
        self._heap = [(v, k) for k, v in self.items()]
        heapify(self._heap)

    def smallest(self):
        heap = self._heap
        v, k = heap[0]
        while k not in self or self[k] != v:
            heappop(heap)
            v, k = heap[0]
        return k

    def get(self):
        heap = self._heap
        v, k = heappop(heap)
        while k not in self or self[k] != v:
            v, k = heappop(heap)
        del self[k]
        return k

    def __setitem__(self, key, val):
        super(priority_dict, self).__setitem__(key, val)

        if len(self._heap) < 2 * len(self):
            heappush(self._heap, (val, key))
        else:
            self._rebuild_heap()

    def setdefault(self, key, val):
        if key not in self:
            self[key] = val
            return val
        return self[key]

    def update(self, *args, **kwargs):
        super(priority_dict, self).update(*args, **kwargs)
        self._rebuild_heap()

    def sorted_iter(self):
        while self:
            yield self.pop_smallest()

def generate_b(n_node, origin, destination):
    '''
    OD start from 1 when displayed or inputted, but start from 0 when stored and calculated.
    '''
    b = np.zeros(n_node)

    r_0 = origin-1
    r_s = destination-1

    b[r_0] = 1
    b[r_s] = -1

    return b.reshape(-1,1), r_0, r_s

def generate_samples(mymap, S):
    '''
    return: N*S matrix
    '''
    rng = np.random.default_rng()
    if mymap.model == "G":
        samples = rng.multivariate_normal(mymap.mu.reshape(-1), mymap.cov, S, method=mymap.decom)
        for i in range(samples.shape[0]):
            for j in range(samples.shape[1]):
                while samples[i][j] <= 0:
                    samples[i][j] = np.random.normal(mymap.mu[j].item(), np.sqrt(mymap.cov[j][j]))

    return samples.T
    
def sort_path_order(path, mymap):
    if type(path) is np.ndarray:
        path = path.tolist()
    sorted_path = []
    node = mymap.r_0
    while node != mymap.r_s:
        for link in path:
            if mymap.M[node,link] == 1:
                sorted_path.append(link)
                node = np.where(mymap.M[:,link]==-1)[0].item()
                path.remove(link)
                break

    return np.array(sorted_path)

def first_path_link(path, mymap):
    if type(path) is np.ndarray:
        path = path.tolist()
    node = mymap.r_0
    for link in path:
        if mymap.M[node,link] == 1:
            sorted_path=[link]
            path.remove(link)
            break
    return np.array(sorted_path+path)

def convert_node2onehot(path, G):
    link_ids = []
    node_pairs=zip(path[0:],path[1:])

    for u,v in node_pairs:
        edge = sorted(G[u][v], key=lambda x:G[u][v][x]['weight'])
        link_ids.append(G[u][v][edge[0]]['index'])

    onehot = np.zeros(G.size())
    onehot[link_ids] = 1
    onehot = onehot.reshape(-1,1)

    return link_ids, onehot

def convert_map2graph(mymap, is_multi=True):
    G = nx.MultiDiGraph() if is_multi else nx.DiGraph()

    for i in range(mymap.M.shape[1]):
        start = np.where(mymap.M[:,i]==1)[0].item()
        end = np.where(mymap.M[:,i]==-1)[0].item()
        G.add_edge(start, end, weight=mymap.mu[i].item(), index=i)

    return G

def find_next_node(mymap, curr_node, link_idx):
    for _, next_node, d in mymap.G.out_edges(curr_node, data=True):
        if d['index'] == link_idx:
            return next_node

def dijkstra(G, start, end, ext_weight=None):
    if not G.has_node(start) or not G.has_node(end):
        return -1, None, None

    cost = {}
    for node in G.nodes():
        cost[node] = float('inf')
    cost[start] = 0
    prev_node = {start: None}
    prev_edge = {start: None}
    PQ = priority_dict(cost)

    while bool(PQ):
        curr_node = PQ.get()

        if curr_node == end:
            break
        
        for _, next_node, d in G.out_edges(curr_node, data=True):
            if next_node in PQ:
                alt = cost[curr_node] + (d['weight'] if ext_weight is None else ext_weight[d['index']].item())
                if alt < cost[next_node]:
                    cost[next_node] = alt
                    prev_node[next_node] = curr_node
                    prev_edge[next_node] = d['index']
                    PQ[next_node] = alt

    if curr_node == end and end in prev_node:
        path_cost = cost[end]
        path = []
        while curr_node != start:
            path.append(prev_edge[curr_node])
            curr_node = prev_node[curr_node]
        path.reverse()

        onehot = np.zeros(G.size())
        onehot[path] = 1
        onehot = onehot.reshape(-1, 1)
        return path_cost, path, onehot
    else:
        return -1, None, None

def path_link2node(mymap, link_path):
    node_path = [mymap.r_0]
    curr_node = mymap.r_0

    for link in link_path:
        next_node = find_next_node(mymap, curr_node, link)
        node_path.append(next_node)
        curr_node = next_node

    return node_path

def path_node2link(mymap, node_path):
    assert is_multigraphical(mymap.G), 'Cannot convert node path to link path on a multigraph'
    link_path = []

    for i in range(len(node_path)-1):
        link_path.append(mymap.G[node_path[i]][node_path[i+1]]['index'])

    return link_path

def k_shortest_paths(mymap, k, weight="weight"):
    return list(islice(nx.shortest_simple_paths(mymap.G, mymap.r_0, mymap.r_s, weight=weight), k))

def t_test(x, y, alternative='greater', alpha=0.05):
    t_stat, double_p = stats.ttest_ind(x,y,equal_var = False)

    if alternative == 'both-sided':
        pval = double_p
    elif alternative == 'greater':
        if t_stat > 0:
            pval = double_p/2.
        else:
            pval = 1.0 - double_p/2.
    elif alternative == 'less':
        if t_stat < 0:
            pval = double_p/2.
        else:
            pval = 1.0 - double_p/2.

    return pval, pval<alpha

def generate_OD_pairs(mymap, n_pair):
    def generate_OD(n_node):
        r_0 = np.random.randint(n_node) + 1
        while not mymap.G.has_node(r_0-1):
            r_0 = np.random.randint(n_node) + 1
        r_s = np.random.randint(n_node) + 1
        while r_s == r_0 or not mymap.G.has_node(r_s-1):
            r_s = np.random.randint(n_node) + 1
        OD = [r_0, r_s]
        return OD

    OD_pairs = []
    count = 0

    while count < n_pair:
        OD = generate_OD(mymap.n_node)
        while OD in OD_pairs or dijkstra(mymap.G, OD[0]-1, OD[1]-1)[0] == -1:
            OD = generate_OD(mymap.n_node)
        OD_pairs.append(OD)
        count += 1

    return OD_pairs
	
def extract_map(map_id, map_dir):
    map_list = ['Simple', 'SiouxFalls', 'Anaheim', 'Winnipeg', 'Chicago_Sketch']

    map_dir += map_list[map_id] + '/'

    raw_map_data = pd.read_csv(map_dir + map_list[map_id] + '_network.csv')

    origins = raw_map_data['From']
    destinations = raw_map_data['To']
    if origins.min() == 0 or destinations.min() == 0:
        origins += 1
        destinations += 1
    n_node = max(origins.max(), destinations.max())
    n_link = raw_map_data.shape[0]

    M = np.zeros((n_node,n_link))
    for i in range(n_link):
        M[origins[i]-1,i] = 1
        M[destinations[i]-1,i] = -1

    mu = np.array(raw_map_data['Cost']).reshape(-1,1)
    cov = np.zeros((n_link, n_link))

    return M, mu, cov