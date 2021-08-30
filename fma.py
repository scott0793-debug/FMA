import numpy as np
import func
import time
from copy import deepcopy
from func import Map
from evaluation import calc_path_prob

class FMA_MAP(Map):
    def __init__(self, model="None"):
        super().__init__(model=model)

    def generate_real_map(self, map_id, map_dir, nu=0.5):
        super().generate_real_map(map_id, map_dir)
        if map_id == 0: 
            var = [5, 3, 2, 3, 2, 3.5, 2, 3.5, 4]
            self.cov = np.diag(var)
        elif map_id == 1:   
            self.cov = np.load(map_dir+'SiouxFalls/variance.npy')
        else:
            self.cov = self.gen_var(nu)
        self.calc_link_moments()

    def gen_var(self, nu):
        mu = self.mu.reshape(-1)
        sigma = nu*mu*np.random.rand(self.n_link)
        var = np.diag(np.square(sigma))
        return var

    def calc_link_moments(self):
        mu = self.mu
        var = np.diag(self.cov).reshape(-1,1)
        self.M1 = mu
        self.M2 = mu*mu + var
        self.M3 = mu*mu*mu + 3*mu*var
        self.M4 = mu*mu*mu*mu + 6*mu*mu*var + 3*var*var
        

class FMA_POLICY():
    def __init__(self, mymap):
        self.map = mymap

    def bad_init_policy(self):   #experiment1
        policy_link = np.zeros(self.map.n_node).astype(int)
        policy_node = np.zeros(self.map.n_node).astype(int)
        policy_link[self.map.r_s] = -1
        policy_node[self.map.r_s] = -1
        policy_link = np.array([0, 3, 5, 6, 7, -1])
        policy_node = np.array([1, 4, 3, 5, 3, -1])
        self.policy_link = policy_link
        self.policy_node = policy_node
        self.PL = self.policy_link2onehot(policy_link)
        self.PN = self.policy_node2onehot(policy_node)
        self.path = self.retrieve_node_path()


    def gen_let_policy(self):
        policy_link = np.zeros(self.map.n_node).astype(int)
        policy_node = np.zeros(self.map.n_node).astype(int)
        policy_link[self.map.r_s] = -1
        policy_node[self.map.r_s] = -1
        for node in self.map.G.nodes:
            if node != self.map.r_s:
                path = func.dijkstra(self.map.G, node, self.map.r_s)[1]
                policy_link[node] = path[0]
                policy_node[node] = func.find_next_node(self.map, node, path[0])

        self.policy_link = policy_link
        self.policy_node = policy_node
        self.PL = self.policy_link2onehot(policy_link)
        self.PN = self.policy_node2onehot(policy_node)
        self.path = self.retrieve_node_path()

    def path2policy(self, link_path, T):
        curr_node = self.map.r_0
        policy_link = np.zeros(self.map.n_node).astype(int)
        policy_node = np.zeros(self.map.n_node).astype(int)
        policy_link[self.map.r_s] = -1
        policy_node[self.map.r_s] = -1

        for link in link_path:
            next_node = func.find_next_node(self.map, curr_node, link)
            policy_link[curr_node] = link
            policy_node[curr_node] = next_node
            curr_node = next_node

        self.policy_link = policy_link
        self.policy_node = policy_node
        self.PL = self.policy_link2onehot(self.policy_link)
        self.PN = self.policy_node2onehot(self.policy_node)
        self.path = self.retrieve_node_path()
        g = self.policy_evaluation(T)

        return g

    def calc_node_moments(self):
        '''
        Here, for the EBE module, we implement a method different from the one in the paper to calculate the 1st-4th moments of each node.
        Instead of approximating the true values iteratively with Extended Bellman Equation,
        we directly calculate them through solving sets of linear equations,
        which describes the relationship between the moments of links(known) and nodes(unknown).
        This relationship IS essentially the deterministic routing policy itself, 
        which is a path connecting the origin and destination.

        We omit this implementation detail in the paper since it is onlt a trick and is trival to the overall target of the EBE module.
        '''
        # PL = self.PL
        # PN = self.PN
        # I_A = np.eye(self.map.n_node) - PN

        path_len = len(self.path)
        PL = self.PL[self.path]
        PN = self.PN[self.path][:, self.path]
        I_A = np.eye(path_len) - PN

        Mij1 = np.dot(PL, self.map.M1)
        b = Mij1
        Mi1 = np.dot(np.linalg.inv(I_A), b)
        Mj1 = np.dot(PN, Mi1)

        Mij2 = np.dot(PL, self.map.M2)
        b = Mij2 + 2*Mij1*Mj1
        Mi2 = np.dot(np.linalg.inv(I_A), b)
        Mj2 = np.dot(PN, Mi2)

        Mij3 = np.dot(PL, self.map.M3)
        b = Mij3 + 3*Mij1*Mj2 + 3*Mij2*Mj1
        Mi3 = np.dot(np.linalg.inv(I_A), b)
        Mj3 = np.dot(PN, Mi3)

        Mij4 = np.dot(PL, self.map.M4)
        b = Mij4 +4*Mij3*Mj1 +6*Mij2*Mj2 + 4*Mij1*Mj3
        Mi4 = np.dot(np.linalg.inv(I_A), b)

        self.Mo1 = Mi1[0].item() #[self.map.r_0]
        self.Mo2 = Mi2[0].item() #[self.map.r_0]
        self.Mo3 = Mi3[0].item() #[self.map.r_0]
        self.Mo4 = Mi4[0].item() #[self.map.r_0]

    def calc_g_value(self, T):
        Mo1 = self.Mo1
        Mo2 = self.Mo2
        Mo3 = self.Mo3
        Mo4 = self.Mo4

        MZ1 = Mo1 - T
        MZ2 = Mo2 - 2*Mo1*T + T**2
        MZ3 = Mo3 - 3*Mo2*T + 3*Mo1*T**2 - T**3
        MZ4 = Mo4 - 4*Mo3*T + 6*Mo2*T**2 - 4*Mo1*T**3 + T**4
        sqrt3 = np.sqrt(3)

        alpha = np.sqrt((MZ4-MZ2**2)/(MZ2-MZ1**2))
        V_min = np.sqrt((sqrt3-1)*MZ2+(7-4*sqrt3)/4*MZ1**2) + (2-sqrt3)*MZ1/2

        cond = 0
        if MZ1 == 0 and MZ4/MZ2**2 >= (3*sqrt3-3)/2:
            g = 1 - (2*sqrt3-3)/MZ4*MZ2**2
            cond = 1
        elif MZ1 == 0 and MZ4/MZ2**2 <= (3*sqrt3-3)/2:
            g = 0.5 + np.sqrt(0.25-1/(3+MZ4/MZ2**2))
            cond = 2
        elif MZ4/MZ2**2 >= MZ2/MZ1**2:
            if MZ1 < 0:
                g = 1 - MZ1**2/MZ2
                cond = 3.1
            else:
                g = 1
                cond = 3.2
        elif MZ4/MZ2**2 < MZ2/MZ1**2 and alpha >= (sqrt3-1)*V_min/2:
            root = np.roots([-MZ1, 3*MZ2, 0, -2*MZ4])
            root = np.real(root[np.isreal(root)])
            root = root[root>0].tolist()
            assert root, 'no v>0'
            sup = np.max([-2*MZ1/v + 3*MZ2/v**2 - MZ4/v**4 for v in root])
            g = 1 - 4/9 * (2*sqrt3 - 3) * sup
            cond = 4
        elif MZ4/MZ2**2 < MZ2/MZ1**2 and alpha <= (sqrt3-1)*V_min/2:
            g = 0.5 * (1 + (alpha+2*MZ1)/np.sqrt(4*MZ2+alpha**2+4*MZ1*alpha))
            cond = 5
            

        print('condition = ' + str(cond))
        return g

    def policy_evaluation(self, T):
        self.calc_node_moments()
        g = self.calc_g_value(T)
        return g

    def policy_update(self, new_link):
        node, next_node, d = new_link
        link = d['index']
        self.policy_node[node] = next_node
        self.policy_link[node] = link
        self.PL = self.policy_link2onehot(self.policy_link)
        self.PN = self.policy_node2onehot(self.policy_node)
        self.path = self.retrieve_node_path()
        self.Mo1 = None
        self.Mo2 = None
        self.Mo3 = None
        self.Mo4 = None

    def policy_node2onehot(self, policy_node):
        policy_node_oh = np.vstack([np.eye(self.map.n_node), np.zeros(self.map.n_node)])
        policy_node_oh = policy_node_oh[policy_node]
        return policy_node_oh

    def policy_link2onehot(self, policy_link):
        policy_link_oh = np.vstack([np.eye(self.map.n_link), np.zeros(self.map.n_link)])
        policy_link_oh = policy_link_oh[policy_link]
        return policy_link_oh
    
    def retrieve_node_path(self, policy_node=None, start=None):
        path = []
        if policy_node is None:
            policy_node = self.policy_node
        curr_node = self.map.r_0 if start is None else start

        while curr_node != -1:
            path.append(curr_node)
            curr_node = policy_node[curr_node]

        return path
    
    def retrieve_link_path(self):
        path = []
        curr_node = self.map.r_0

        while curr_node != self.map.r_s:
            path.append(self.policy_link[curr_node])
            curr_node = self.policy_node[curr_node]

        return path

class FMA():
    def __init__(self, mymap, T, K=100):
        self.map = mymap
        self.T = T
        self.K = K

    def policy_iteration(self):
        t1 = time.perf_counter()

        self.policy = FMA_POLICY(self.map)
        self.policy.gen_let_policy()
        # self.policy.bad_init_policy() #experiment1
        self.explorered = [self.policy.path]

        g_star = self.policy.policy_evaluation(self.T)
        print("LET path:" )
        print(self.policy.path)
        print("g_prime = " + str(g_star))

        for cnt in range(self.K):
            print("\ncurrent iteration: " + str(cnt))
            new_policy = self.gen_new_policy()
            if new_policy is None:
                print("Error: cannot find new random policy.")
                break
            g_prime = new_policy.policy_evaluation(self.T)

            print(new_policy.path)
            print("g_prime = " + str(g_prime))

            if g_prime < g_star:
                self.policy = new_policy
                g_star = g_prime
                print("improved!")

        t_delta = time.perf_counter() - t1 
        
        final_link_path = self.policy.retrieve_link_path()
        print("\nFinal:")
        print("node_path: " + str(self.policy.path))
        print("link_path: " + str(final_link_path))
        print("g_star = " + str(g_star))
        prob = calc_path_prob(final_link_path, self.map, self.T)
        return prob, g_star, t_delta, self.policy.path, final_link_path

    def gen_new_policy(self):
        '''
        policy variation
        '''
        new_path = self.policy.path
        count = 0

        while new_path in self.explorered:
            count += 1
            if count >= 100:
                return None
            random_node_idx = np.random.choice(len(self.policy.path)-1)
            random_node = self.policy.path[random_node_idx]

            link_list = list(self.map.G.out_edges(random_node, data=True))
            random_link_idx = np.random.choice(len(link_list))
            random_link = link_list[random_link_idx]
            random_next_node = random_link[1]

            if random_next_node == self.policy.path[random_node_idx+1]:
                continue
            if func.dijkstra(self.map.G, random_next_node, self.map.r_s)[0] == -1:
                continue
            if self.is_cyclic(self.policy.path[:random_node_idx+1], self.policy.retrieve_node_path(start=random_next_node)):
                continue

            new_policy_node = self.policy.policy_node.copy()
            new_policy_node[random_node] = random_next_node
            new_path = self.policy.retrieve_node_path(policy_node=new_policy_node)
        
        self.explorered.append(new_path)
        new_policy = deepcopy(self.policy)
        new_policy.policy_update(random_link)
        return new_policy

    def is_cyclic(self, old_path, new_path):
        '''
        Check cyclic
        '''
        old_path = set(old_path)
        new_path = set(new_path)
        if old_path.intersection(new_path):
            return True
        else:
            return False

def calc_path_g(path, mymap, T):
    policy = FMA_POLICY(mymap)
    return policy.path2policy(path, T)

def LET_path(mymap, T):
    t1 = time.perf_counter()
    _, d_path, _ = func.dijkstra(mymap.G, mymap.r_0, mymap.r_s)
    t_delta = time.perf_counter() - t1 
    g = calc_path_g(d_path, mymap, T)
    g_delta = 1-g
    prob = calc_path_prob(d_path, mymap, T)
    return prob, g, 0, t_delta, g_delta