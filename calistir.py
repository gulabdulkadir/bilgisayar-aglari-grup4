# Q LEARNING PARAMETRELERI
from genetik_algoritma import genetic_algorithm
import tabusearch
EPSILON = 5.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.998
MAX_STEPS = 50
EPISODE = 1500
ALPHA = 0.9
GAMMA = 0.9
############################

import network_model
import q_learning

if __name__ == "__main__":
    talepler = network_model.AgOrtami().talep_listesini_getir()
    deneme_talep = talepler[12]
    KAYNAK = deneme_talep['kaynak']
    HEDEF = deneme_talep['hedef']
    GEREKLI_BANDWIDTH = deneme_talep['miktar']

    # Q LEARNING BOLUMU
    ##################################################
    print("Q Learning Çağırılıyor...")
    # topoloji olustur filtrele
    ag_topolojisi = network_model.AgOrtami()  # ag topolojisini kur
    ag_topolojisi.verileri_yukle_ve_agi_kur()
    env = q_learning.NetworkRoutingEnv(ag_topolojisi.graf, KAYNAK, HEDEF,
                                       GEREKLI_BANDWIDTH)  # filtrelenmis environment oluşturduk
    q_table = q_learning.q_learning_egit(HEDEF=HEDEF, KAYNAK=KAYNAK, env=env, episode=EPISODE, epsilon=EPSILON,
                                         epsilon_min=EPSILON_MIN, epsilon_decay=EPSILON_DECAY,
                                         alpha=ALPHA, gamma=GAMMA, max_steps=MAX_STEPS)
    q_learning_path = q_learning.q_learning_calistir(HEDEF, KAYNAK, env, q_table, MAX_STEPS)
    q_toplam_skor, q_toplam_gecikme, q_toplam_guven_maliyet, q_toplam_kaynak_maliyet = ag_topolojisi.yol_maliyeti_hesapla(
        q_learning_path, GEREKLI_BANDWIDTH)
    print(f"Q Learning Toplam Skor: {q_toplam_skor}")
    print(f"Q Learning Toplam Gecikme: {q_toplam_gecikme} ms")
    print(f"Q Learning Toplam Guven Maliyeti: {q_toplam_guven_maliyet}")
    print(f"Q Learning Toplam Kaynak Maliyeti: {q_toplam_kaynak_maliyet}")
    print("====================")

    #################################################

#TABU SEARCH BOLUMU


    ortam = network_model.AgOrtami()
    ortam.verileri_yukle_ve_agi_kur()


    S = KAYNAK
    D = HEDEF
    bw = GEREKLI_BANDWIDTH

    best_path, best_cost = genetic_algorithm(
        ortam=ortam,
        S=S,
        D=D,
        bw=bw,
        pop_size=50,
        generations=100
    )

    if best_path:
        print("Genetik algoritma başarıyla tamamlandı.")
        print(f"En iyi yol bulundu ({len(best_path)} düğüm):")
        print(f"  {best_path}")
        print(f"Toplam Fitness : {best_cost:.4f}")

        _, delay, reliability, resource_cost = ortam.yol_maliyeti_hesapla(
            best_path, istenen_bw=bw
        )
        print(f"Gecikme        : {delay:.2f} ms")
        print(f"Güvenilirlik  : {reliability:.4f}")
        print(f"Kaynak Maliyeti: {resource_cost:.2f}")
    else:
        print("Uygun bir yol bulunamadı.")
# --- TABU SEARCH BÖLÜMÜ ---
print("\n" + "="*20)
print("TABU SEARCH SONUÇLARI")
print("="*20)

# Algoritmayı çalıştır
tabu_path = tabusearch.tabu_search(ag_topolojisi, KAYNAK, HEDEF, GEREKLI_BANDWIDTH)

if tabu_path:
    # Senin network_model.py içindeki hesaplamayı kullanıyoruz
    t_skor, t_gec, t_guv_mal, t_kaynak = ag_topolojisi.yol_maliyeti_hesapla(tabu_path, GEREKLI_BANDWIDTH)
    
    import math
    guvenilirlik_yuzde = math.exp(-t_guv_mal) * 100 # Güvenilirliği yüzdeye çevir

    print(f"Rota: {tabu_path}")
    print(f"Toplam Skor (Fitness): {t_skor:.4f}")
    print(f"Gecikme: {t_gec:.2f} ms")
    print(f"Güvenilirlik: %{guvenilirlik_yuzde:.4f}")
    print(f"Kaynak Maliyeti: {t_kaynak:.2f}")
else:
    print("HATA: Tabu Search bir yol bulamadı!")
print("="*20)
#########################################################################################################################
