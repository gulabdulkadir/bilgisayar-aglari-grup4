import network_model
import math
import numpy as np
import random
from tqdm import tqdm
import time
import genetik_algoritma

# ==========================================
# SENİN SABİTLERİN (DOKUNULMADI)
# ==========================================
W_DELAY = 0.33
W_RELIABILITY = 0.33
W_BANDWIDTH = 0.33

MAX_DELAY = 17.0
MIN_DELAY = 3.5
MIN_RELIABILITY = 0.002
MAX_RELIABILITY = 0.103
MIN_BANDWIDTH = 1.0
MAX_BANDWIDTH = 10.0

# ==========================================
# YARDIMCI FONKSİYONLAR VE SINIFLAR (DOKUNULMADI)
# ==========================================
def topoloji_filtrele(G, required_bandwidth):
    """
    Orijinal grafiği (G) bozmadan, sadece bant genişliği yeten
    linklerden oluşan geçici bir 'Görünüm' (View) oluşturur.
    """
    valid_edges = [
        (u, v) for u, v, data in G.edges(data=True) 
        if data['bant_genisligi'] >= required_bandwidth
        ]
    
    filtrelenmis_graf = G.edge_subgraph(valid_edges)
    return filtrelenmis_graf

    
class NetworkRoutingEnv():
    def __init__(self, G, source, target, bandwidth_demand, weights=(W_DELAY, W_RELIABILITY, W_BANDWIDTH)):
        
        if G.is_directed():
            G = G.to_undirected()
        
        self.G = topoloji_filtrele(G, required_bandwidth = bandwidth_demand)
        self.source = source
        self.bandwidth_demand = bandwidth_demand
        self.weights = weights 
        self.target = target      
        self.nodes = list(G.nodes)              
        self.state = self.source
        self.path_history = [self.source]
        self.visited = set([self.source])

    def reset(self):
        """Her bölüm (episode) başında ajanı başa sarar."""
        self.state = self.source
        self.path_history = [self.source]
        self.visited = set([self.source])
        return self.state
    

    def calculate_reward(self, action, HEDEF, KAYNAK ):
        
        if action == HEDEF :
            total_delay = self.G.edges[self.state, action]['gecikme']
        else:
            total_delay = (self.G.nodes[action]['islem_suresi'] + self.G.edges[self.state, action]['gecikme'])
        
        total_reliability = -math.log(self.G.edges[self.state, action]['guvenilirlik']) + (-math.log(self.G.nodes[action]['guvenilirlik']))
        bw_cost = 1000/(self.G.edges[self.state, action]['bant_genisligi'] + 1e-5)
        
        reward = W_DELAY * total_delay + W_RELIABILITY * total_reliability + W_BANDWIDTH * bw_cost 
        return -reward


    def step(self, action, HEDEF, KAYNAK):
        """
        Ajanın bir düğümden diğerine geçme hamlesi.
        action: Gitmek istenilen düğüm ID'si.
        """
        done = False
        
        current_node = self.state
        next_node = action
        
        if next_node in self.visited:
            return current_node, -1000, True
        
        reward = self.calculate_reward(action, HEDEF, KAYNAK)
        
        self.state = next_node
        self.path_history.append(self.state)
        
        self.visited.add(self.state)
        
        if self.state == self.target:
            reward += 5
            done = True
        
        return self.state, reward, done
    
def q_learning_egit(HEDEF, KAYNAK, env, episode, epsilon, epsilon_min, epsilon_decay, alpha, gamma, max_steps):
    
    env.reset() 
    
    # Node ID'leri sıralı gitmeyebilir, garantiye almak için max ID'ye göre tablo
    all_nodes = list(env.G.nodes)
    max_id = max(all_nodes) if all_nodes else 250
    state_space = max_id + 1
    action_space = max_id + 1
    
    q_table = np.full((state_space, action_space), 0.00) 
    
    # =================================================================
    # <<< 1. EKLEME: ÖN BELLEK OLUŞTURMA (DÖNGÜDEN ÖNCE) >>>
    # =================================================================
    print(">> Komşular önbelleğe alınıyor (Optimization)...")
    # Bu satır, networkx kütüphanesine sürekli sormak yerine, komşuları hafızaya alır. 
    adjacency_cache = {node: list(env.G.neighbors(node)) for node in env.G.nodes}
    # =================================================================

    print(">> Q Learning Eğitim Başlıyor...")
    t1 = time.time()
    for i in tqdm(range(0, episode)):
        
        state = env.reset()
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            
            # =============================================================
            # <<< 2. DEĞİŞİKLİK: KOMŞULARI CACHE'DEN ÇEKME >>>
            # Eski: neighbors = list(env.G.neighbors(state))
            # Yeni (Çok daha hızlı):
            neighbors = adjacency_cache.get(state, [])
            # =============================================================
            
            if not neighbors:
                action = state


            elif random.uniform(0, 1) < 0.5 and HEDEF in neighbors: 
                action = HEDEF
                
            elif random.uniform(0, 1) < epsilon: 
                action = random.choice(neighbors)
                
            else:   
                
                if not neighbors:
                    action = state
                else:
                    neighbor_q_values = [q_table[state, n] for n in neighbors]
                    
                    # Tie-breaking eklemek istersen burayı güncelleyebilirsin
                    best_index_local = np.argmax(neighbor_q_values) 
                    
                    action = neighbors[best_index_local] 
                        
            next_state, reward, done = env.step(action, HEDEF, KAYNAK)
   
            # --- 3. GELECEĞİ HESAPLA  ---
            
            # =============================================================
            # <<< 3. DEĞİŞİKLİK: GELECEK KOMŞULARI CACHE'DEN ÇEKME >>>
            # Eski: next_neighbors = list(env.G.neighbors(next_state))
            # Yeni:
            next_neighbors = adjacency_cache.get(next_state, [])
            # =============================================================
            
            if not next_neighbors:
                max_future_q = -1000.0 
            else:
                future_qs = [q_table[next_state, n] for n in next_neighbors]
                max_future_q = max(future_qs) 
            
            # --- 4. GÜNCELLEME ---
            current_q = q_table[state, action]
            new_q = current_q + alpha * (reward + gamma * max_future_q - current_q)
            q_table[state, action] = new_q
            
            state = next_state      
            steps += 1
            
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
            
    t2 = time.time()
    print (f"\nQ Learning Eğitim Süresi: {t2 - t1}")
    print("Q Learning Training finished")
    
    return q_table



def q_learning_calistir(HEDEF, KAYNAK, env, q_table, max_steps):
    print(">> Q Learning En İyi Rota Hesaplanıyor...")
    curr = env.reset()
    path = [curr] 
    
    for _ in range(max_steps):
            neighbors = list(env.G.neighbors(curr))
        
            if not neighbors:
                print("❌ Çıkmaz sokak.")
                break
            
            neighbor_q_values = [q_table[curr, n] for n in neighbors]
            best_idx = np.argmax(neighbor_q_values)
            action = neighbors[best_idx]
        
            next_node, r, d = env.step(action, HEDEF, KAYNAK)
            path.append(next_node)
            curr = next_node
        
            if d: 
                if curr == HEDEF:
                    print(f"Q Learning🏆 HEDEF BULUNDU! Rota: {path}")                    
                else:
                    print("❌ BAŞARISIZ! Ajan bir döngüye girdi ve tur sonlandırıldı.")
                break
    else:
            print("❌ Hedefe ulaşılamadı.")
    return path

# ==========================================
# LOCAL TEST BLOK (AYNEN KORUNDU)
# ==========================================
if __name__ == "__main__":
    #parametreler
    epsilon = 2.0  #keşif için sınırlama değişkenleri 0.999 öğrenme ile başlıyor
    epsilon_min = 0.1 #her seferinde 0.998 ile çarpıyor 0.1 olunca daha fazla azaltmıyor
    epsilon_decay = 0.998
    max_steps = 50 #sonsuz döngü olmamamsı için maximum adım sayisi belirledik
    episode = 1500
    alpha = 0.5 # öğrenme oranı (learning rate)
    gamma = 0.9 # discount rate
    
    talepler = network_model.AgOrtami().talep_listesini_getir()
    
    deneme_talep = talepler[1]
    KAYNAK = deneme_talep['kaynak']
    HEDEF = deneme_talep['hedef']
    GEREKLI_BANDWIDTH = deneme_talep['miktar']
    #topoloji olustur filtrele
    ag_topolojisi = network_model.AgOrtami() #ag topolojisini kur
    ag_topolojisi.verileri_yukle_ve_agi_kur() #exceldeki verileri yukle
    env = NetworkRoutingEnv(ag_topolojisi.graf, KAYNAK, HEDEF, GEREKLI_BANDWIDTH) #filtrelenmis environment oluşturduk
    
    #q_learning_egit
    q_table = q_learning_egit(HEDEF, KAYNAK, env = env, episode = episode, epsilon = epsilon,
                              epsilon_min = epsilon_min, epsilon_decay = epsilon_decay,
                              alpha = alpha, gamma = gamma, max_steps = max_steps)

    #q_learning_calistir                   
    path = q_learning_calistir(HEDEF, KAYNAK, env, q_table, max_steps)
    print (f"Yol Maliyeti: {ag_topolojisi.yol_maliyeti_hesapla(path, GEREKLI_BANDWIDTH)}")
    genetik_path = genetik_algoritma.genetic_algorithm(ag_topolojisi, KAYNAK, HEDEF, GEREKLI_BANDWIDTH)
    print(f"deneme yol maliyeti {genetik_path}")