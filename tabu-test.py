import matplotlib.pyplot as plt
import tabusearch
import network_model

# 1. Hazırlık: Ağı ve talebi oluştur
ag = network_model.AgOrtami()
ag.verileri_yukle_ve_agi_kur()
talepler = ag.talep_listesini_getir()
talep = talepler[0] # Test için ilk talebi kullanıyoruz

# 2. Test Parametrelerini Belirle (Grid Search Alanı)
iterasyon_denemeleri = [10, 20, 50, 80, 100, 150]
tabu_boyutu_denemeleri = [5, 10, 15, 20, 25, 30]

iter_sonuclari = []
tabu_sonuclari = []

print("🚀 Parametre testleri başlatılıyor...")

# --- TEST 1: İterasyon Sayısının Etkisi ---
for i in iterasyon_denemeleri:
    yol = tabusearch.tabu_search(ag, talep['kaynak'], talep['hedef'], talep['miktar'], iterasyon=i)
    skor, _, _, _ = ag.yol_maliyeti_hesapla(yol, talep['miktar'])
    iter_sonuclari.append(skor)

# --- TEST 2: Tabu Listesi Boyutunun Etkisi ---
for b in tabu_boyutu_denemeleri:
    yol = tabusearch.tabu_search(ag, talep['kaynak'], talep['hedef'], talep['miktar'], tabu_boyutu=b)
    skor, _, _, _ = ag.yol_maliyeti_hesapla(yol, talep['miktar'])
    tabu_sonuclari.append(skor)

# --- 3. GÖRSELLEŞTİRME 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Sol Grafik: İterasyon
ax1.scatter(iterasyon_denemeleri, iter_sonuclari, color='blue', s=100)
ax1.plot(iterasyon_denemeleri, iter_sonuclari, color='blue', linestyle='--', alpha=0.5)
ax1.set_title('İterasyon Sayısı vs Maliyet')
ax1.set_xlabel('İterasyon')
ax1.set_ylabel('Toplam Maliyet (Fitness)')
ax1.grid(True)

# Sağ Grafik: Tabu Boyutu
ax2.scatter(tabu_boyutu_denemeleri, tabu_sonuclari, color='red', s=100)
ax2.plot(tabu_boyutu_denemeleri, tabu_sonuclari, color='red', linestyle='--', alpha=0.5)
ax2.set_title('Tabu Boyutu vs Maliyet')
ax2.set_xlabel('Tabu Listesi Boyutu')
ax2.set_ylabel('Toplam Maliyet (Fitness)')
ax2.grid(True)

en_iyi = min(min(iter_sonuclari), min(tabu_sonuclari))
plt.suptitle(f"Tabu Search Parametre Analizi (En İyi Skor: {en_iyi:.4f})", fontsize=14)
plt.tight_layout()

# Grafiği kaydet
plt.savefig('tabu_parametre_test.png')
print("✅ Test tamamlandı! 'tabu_parametre_test.png' dosyası oluşturuldu.")
plt.show()