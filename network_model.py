import pandas as pd
import networkx as nx
import math

# --- SABİTLER ---
MAKS_BANT_GENISLIGI = 1000.0 # 1000 Mbps = 1 Gbps (Dokümandaki formül için)

def agi_baslat():
    """CSV dosyalarını okur ve ağ yapısını oluşturur."""
    try:
        # CSV dosyalarınızı okuyoruz (Noktalı virgül ';' ayracını kullanıyoruz)
        dugum_verisi = pd.read_csv("BSM307_317_Guz2025_TermProject_NodeData.csv", sep=';')
        baglanti_verisi = pd.read_csv("BSM307_317_Guz2025_TermProject_EdgeData.csv", sep=';')
    except Exception as e:
        print(f"HATA: Dosyalar okunamadı! Lütfen dosya isimlerini kontrol edin: {e}")
        return None

    G = nx.Graph()

    # Düğümleri (Node) ekliyoruz (Gecikme ve Güvenilirlik özellikleriyle)
    for _, satir in dugum_verisi.iterrows():
        G.add_node(int(satir['Node_ID']), 
                   islem_gecikmesi=satir['ProcessingDelay'],
                   guvenilirlik=satir['NodeReliability'])

    # Bağlantıları (Edge) ekliyoruz (Gecikme, Bant Genişliği ve Güvenilirlik ile)
    for _, satir in baglanti_verisi.iterrows():
        u, v = int(satir['Source']), int(satir['Target'])
        # Mbps değerini Gbps'e çeviriyoruz
        bant_gbps = satir['Bandwidth'] / 1000.0
        
        G.add_edge(u, v, 
                   baglanti_gecikmesi=satir['LinkDelay'],
                   bant_genisligi=satir['Bandwidth'],
                   bant_gbps=bant_gbps,
                   guvenilirlik=satir['LinkReliability'])
    
    print(f"Ağ başarıyla kuruldu: {G.number_of_nodes()} düğüm ve {G.number_of_edges()} bağlantı yüklendi.")
    return G

def metrikleri_hesapla(graf, yol):
    """Bir yolun (Path) tüm QoS metriklerini dokümandaki formüllere göre hesaplar."""
    toplam_gecikme = 0.0
    guvenilirlik_maliyeti = 0.0
    kaynak_maliyeti = 0.0

    # 1. Düğüm temelli metrikler (Logaritmik güvenilirlik ve işlem gecikmesi)
    for dugum in yol:
        ozellik = graf.nodes[dugum]
        # Güvenilirlik Maliyeti Formülü: -log(Reliability)
        guvenilirlik_maliyeti += -math.log(ozellik['guvenilirlik'])
        # Kaynak ve hedef hariç, ara düğümlerin işlem gecikmesini ekle
        if dugum != yol[0] and dugum != yol[-1]:
            toplam_gecikme += ozellik['islem_gecikmesi']

    # 2. Bağlantı temelli metrikler (Gecikme, Güvenilirlik ve Kaynak Kullanımı)
    for i, j in zip(yol[:-1], yol[1:]):
        kenar = graf.get_edge_data(i, j)
        toplam_gecikme += kenar['baglanti_gecikmesi']
        guvenilirlik_maliyeti += -math.log(kenar['guvenilirlik'])
        # Kaynak Maliyeti Formülü: 1 Gbps / Bant Genişliği (Gbps)
        kaynak_maliyeti += (1.0 / kenar['bant_gbps'])

    return {
        'gecikme': toplam_gecikme,
        'guvenilirlik_maliyeti': guvenilirlik_maliyeti,
        'kaynak_maliyeti': kaynak_maliyeti
    }

def toplam_maliyeti_getir(graf, yol, w_gecikme, w_guvenilirlik, w_kaynak):
    """Ağırlıklı Toplam Yöntemi ile yolun toplam maliyet puanını hesaplar."""
    m = metrikleri_hesapla(graf, yol)
    # Fitness (Uygunluk) veya Cost (Maliyet) fonksiyonu
    skor = (w_gecikme * m['gecikme'] + 
            w_guvenilirlik * m['guvenilirlik_maliyeti'] + 
            w_kaynak * m['kaynak_maliyeti'])
    return skor