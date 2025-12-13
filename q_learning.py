import network_model
import math
import numpy as np
import random
from tqdm import tqdm
GEREKLI_BANDWIDTH = 100
KAYNAK = 0
HEDEF = 114
MAX_DELAY = 17.0
MIN_DELAY = 3.5
MIN_RELIABILITY = 0.002
MAX_RELIABILITY = 0.103
MIN_BANDWIDTH = 1.0
MAX_BANDWIDTH = 10.0

def topoloji_filtrele(G, required_bandwidth):
        """
        Orijinal grafiği (G) bozmadan, sadece bant genişliği yeten
        linklerden oluşan geçici bir 'Görünüm' (View) oluşturur.
        """
        # List comprehension ile uygun kenarları seçiyoruz
        #bandwidth'i saglamayan edge'ler graftan siliniyor ki qlearningte daha az arama yapıp maliyeti düşürelim
        valid_edges = [
            (u, v) for u, v, data in G.edges(data=True) 
            if data['bant_genisligi'] >= required_bandwidth
            ]
    
        # Sadece bu kenarlara sahip yeni bir graf görünümü oluştur
        # edge_subgraph: Sadece belirtilen kenarları içeren bir alt graf döner
        filtrelenmis_graf = G.edge_subgraph(valid_edges)
    
        return filtrelenmis_graf

    
class NetworkRoutingEnv():
    def __init__(self, G, source, target, bandwidth_demand, weights=(0.33, 0.33, 0.33)):
        self.G = topoloji_filtrele(G, required_bandwidth = GEREKLI_BANDWIDTH)
        self.source = source
        self.bandwidth_demand = bandwidth_demand
        self.weights = weights # (W_delay, W_reliability, W_resource)
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
    

    def calculate_reward(self, action ):
        
        total_delay = (self.G.nodes[action]['islem_suresi'] + self.G.edges[self.state, action]['gecikme'])
        total_reliability = -math.log(self.G.edges[self.state, action]['guvenilirlik']) + (-math.log(self.G.nodes[action]['guvenilirlik']))
        bw_cost = 1000/(self.G.edges[self.state, action]['bant_genisligi'] + 1e-5)
        
        norm_delay = (total_delay - MIN_DELAY) / (MAX_DELAY - MIN_DELAY)
        norm_reliability = (total_reliability - MIN_RELIABILITY) / (MAX_RELIABILITY - MIN_RELIABILITY)
        norm_bandwidth = (bw_cost - MIN_BANDWIDTH) / (MAX_BANDWIDTH - MIN_BANDWIDTH)
        
        reward = (self.weights[0] * norm_delay) + (self.weights[1] * norm_reliability) + (self.weights[2] * norm_bandwidth)
        
        return -reward


    def step(self, action):
        """
        Ajanın bir düğümden diğerine geçme hamlesi.
        action: Gitmek istenilen düğüm ID'si.
        """
        done = False
        
        current_node = self.state
        next_node = action
        
        neighbors = list(self.G.neighbors(current_node))
        
        if next_node not in neighbors:
            return current_node, -100000, False ##burasi calismayacak zaten actionu komsulardan sectik
        
        if next_node in self.visited:
            return current_node, -10000.0, True
        
        reward = self.calculate_reward(action)
        
        self.state = next_node
        self.path_history.append(self.state)
        
        self.visited.add(self.state)
        
        if self.state == self.target:
            reward += 100000
            done = True
        
        return self.state, reward, done


if __name__ == "__main__":
    
    ag_topolojisi = network_model.AgOrtami() #ag topolojisini kur
    ag_topolojisi.verileri_yukle_ve_agi_kur() #exceldeki verileri yukle
    env = NetworkRoutingEnv(ag_topolojisi.graf, KAYNAK, HEDEF, GEREKLI_BANDWIDTH) #filtrelenmis environment oluşturduk
    
    env.reset() #ajanımızı kaynak konumuna gönderdik (resetledik)
    
    action_space = len(ag_topolojisi.graf.nodes) 
    state_space = len(ag_topolojisi.graf.nodes) #state: şu anki konum, action: hareket edebileceği konum q_table için oluşturduk
    
    q_table = np.full((state_space, action_space), -1000.0) #-1000 er puanla her hamleye kötümser başlangıç
    
    alpha = 0.9 # öğrenme oranı (learning rate)
    gamma = 0.9 # discount rate
    max_steps = 250 #sonsuz döngü olmamamsı için maximum adım sayisi belirledik
    
    epsilon = 0.9  #keşif için sınırlama değişkenleri 0.9 öğrenme ile başlıyor
    epsilon_min = 0.1 #her seferinde 0.998 ile çarpıyor 0.1 olunca daha fazla azaltmıyor
    epsilon_decay = 0.998 
    
    print(">> Eğitim Başlıyor...")
    
    for i in tqdm(range(1, 500)):
        
        state = env.reset()
        done = False
        steps = 0
        episode_reward = 0 # Bu turdaki toplam puanı tutar
        
        while not done and steps < max_steps:
            neighbors = list(env.G.neighbors(state))
            
            if not neighbors:
                action = state
                
            elif random.uniform(0, 1) < epsilon:
                action = random.choice(neighbors)
                
            else:   
                
                if not neighbors:
                    action = state
                else:
                    neighbor_q_values = [q_table[state, n] for n in neighbors]
                    
                    best_index_local = np.argmax(neighbor_q_values)
                    
                    action = neighbors[best_index_local]
                    
                       
            next_state, reward, done = env.step(action)
#####    
           # --- 3. GELECEĞİ HESAPLA (DÜZELTME 2: FUTURE MASKING GERİ GELDİ) ---
            # np.max yerine sadece gelecekteki komşulara bakıyoruz
            next_neighbors = list(env.G.neighbors(next_state))
            
            if not next_neighbors:
                max_future_q = -1000.0 # Veya tablonun taban puanı
            else:
                # Gelecekteki düğümün sadece gerçek komşularının puanlarını al
                future_qs = [q_table[next_state, n] for n in next_neighbors]
                max_future_q = max(future_qs)
            
            # --- 4. GÜNCELLEME ---
            current_q = q_table[state, action]
            # Formülde np.max yerine max_future_q kullanıyoruz
            new_q = current_q + alpha * (reward + gamma * max_future_q - current_q)
            q_table[state, action] = new_q
            
            state = next_state      
            steps += 1
            
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
    
    print("Training finished")
 
    print(">> 2. En İyi Rota Hesaplanıyor...")
    curr = env.reset()
    path = [curr] # Rotayı tutacak liste
    
    for _ in range(max_steps):
            # ÖNEMLİ: Test ederken de maskeleme yapmalıyız
            neighbors = list(env.G.neighbors(curr))
        
            if not neighbors:
                print("❌ Çıkmaz sokak.")
                break
            
            # Sadece komşuların Q değerlerini çek
            neighbor_q_values = [q_table[curr, n] for n in neighbors]
            best_idx = np.argmax(neighbor_q_values)
            action = neighbors[best_idx]
        
            next_node, r, d = env.step(action)
            path.append(next_node)
            curr = next_node
        
            if d: 
                if curr == HEDEF:
                    
                    print(f"🏆 HEDEF BULUNDU! Rota: {path}")                    
                else:
                    print("❌ BAŞARISIZ! Ajan bir döngüye girdi ve tur sonlandırıldı.")
                break
    else:
            print("❌ Hedefe ulaşılamadı.")
            
            
    print (f"Yol Maliyeti: {ag_topolojisi.yol_maliyeti_hesapla(path, GEREKLI_BANDWIDTH)}")
    
    
    
    