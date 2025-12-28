#Q LEARNING PARAMETRELERI
EPSILON = 20.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.998
MAX_STEPS = 50
EPISODE = 500
ALPHA = 0.9
GAMMA = 0.9
############################

import network_model
import q_learning

if __name__ == "__main__":
    talepler = network_model.AgOrtami().talep_listesini_getir()
    deneme_talep = talepler[0]
    KAYNAK = deneme_talep['kaynak']
    HEDEF = deneme_talep['hedef']
    GEREKLI_BANDWIDTH = deneme_talep['miktar']
    
    #Q LEARNING BOLUMU
    ##################################################
    print("Q Learning Çağırılıyor...")
    #topoloji olustur filtrele
    ag_topolojisi = network_model.AgOrtami() #ag topolojisini kur
    ag_topolojisi.verileri_yukle_ve_agi_kur()
    env = q_learning.NetworkRoutingEnv(ag_topolojisi.graf, KAYNAK, HEDEF, GEREKLI_BANDWIDTH) #filtrelenmis environment oluşturduk
    q_table = q_learning.q_learning_egit(HEDEF = HEDEF, KAYNAK= KAYNAK, env = env, episode = EPISODE, epsilon = EPSILON,
                          epsilon_min = EPSILON_MIN, epsilon_decay = EPSILON_DECAY,
                          alpha = ALPHA, gamma = GAMMA, max_steps = MAX_STEPS)
    q_learning_path = q_learning.q_learning_calistir(HEDEF, KAYNAK, env, q_table, MAX_STEPS)
    q_toplam_skor, q_toplam_gecikme, q_toplam_guven_maliyet, q_toplam_kaynak_maliyet = ag_topolojisi.yol_maliyeti_hesapla(q_learning_path, GEREKLI_BANDWIDTH)
    print(f"Q Learning Toplam Skor: {q_toplam_skor}")
    print(f"Q Learning Toplam Gecikme: {q_toplam_gecikme} ms")
    print(f"Q Learning Toplam Guven Maliyeti: {q_toplam_guven_maliyet}")
    print(f"Q Learning Toplam Kaynak Maliyeti: {q_toplam_kaynak_maliyet}")
    print("====================")
    
    #################################################
    
    

