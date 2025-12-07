import networkx as nx  # Ağ topolojisini (Graf) oluşturmak ve düğüm/bağlantı ilişkilerini yönetmek için gerekli.
import math            # PDF'teki Güvenilirlik hesabında -log (logaritma) işlemi yapmak için gerekli.
import pandas as pd    # Verilen CSV dosyalarını (Node, Edge, Demand) okuyup tablo olarak işlemek için gerekli.

# =============================================================================
# AYARLAR VE SABİTLER
# =============================================================================
# Kodun tamamında kullanılacak dosya isimlerini burada sabitliyoruz (Değişiklik gerekirse tek yerden yapılır).
DUGUM_DOSYASI = "BSM307_317_Guz2025_TermProject_NodeData.csv"    # Düğüm özelliklerini (işlem süresi, güvenilirlik) tutan dosya.
KENAR_DOSYASI = "BSM307_317_Guz2025_TermProject_EdgeData.csv"    # Bağlantı özelliklerini (hız, gecikme, maliyet) tutan dosya.
TALEP_DOSYASI = "BSM307_317_Guz2025_TermProject_DemandData.csv"  # Kimden kime ne kadar veri gideceğini belirten talep dosyası.


class AgOrtami:
    """
    Bu sınıf, hocanın verdiği CSV dosyalarından verileri okuyarak
    gerçek ağ topolojisini oluşturur ve yönetir.
    """

    def __init__(self):
        # Boş bir graf oluşturuyoruz.
        self.graf = nx.Graph()

        # Optimizasyon Ağırlıkları (Fitness Hesaplama Katsayıları)
        # PDF'te "Toplamları 1 olmalı" kuralı vardır.
        # Başlangıçta hepsine eşit önem vermek için (1/3) yaklaşık değerler girdik.
        # Toplamın tam 1.00 olması için sonuncuyu 0.34 yaptık (0.33 + 0.33 + 0.34 = 1.0).
        self.w_gecikme = 0.33  # Gecikme (Delay) önemi
        self.w_guvenilirlik = 0.33  # Güvenilirlik (Reliability) önemi
        self.w_kaynak = 0.34  # Kaynak (Bandwidth) maliyeti önemi
    def verileri_yukle_ve_agi_kur(self):
        """
        Bu fonksiyon CSV dosyalarını okur ve NetworkX grafiğine dönüştürür.
        Rastgele üretim YERİNE bu fonksiyon kullanılır.
        """
        print("📂 Veriler okunuyor ve ağ oluşturuluyor...")

        # ---------------------------------------------------------
        # ADIM 1: DÜĞÜM (NODE) VERİLERİNİ YÜKLEME
        # ---------------------------------------------------------
        try:
            # Pandas kütüphanesi ile CSV dosyasını okuyup 'DataFrame' (akıllı tablo) formatına çeviriyoruz.
            # Parametrelerin Anlamları:
            # 1. DUGUM_DOSYASI: Okunacak dosyanın adı.
            # 2. sep=';'      : Sütunların noktalı virgül ile ayrıldığını belirtir (Normalde virgüldür).
            # 3. decimal=','  : Sayılardaki ondalık ayracının virgül olduğunu belirtir (Python normalde nokta '.' bekler).
            #                   Bunu yapmazsak Python sayıları 'metin' sanar ve matematik işlemi yapamayız.
            dugum_verisi = pd.read_csv(DUGUM_DOSYASI, sep=';', decimal=',')

            # -------------------------------------------------------------------------
            # DÖNGÜ BAŞLANGICI
            # -------------------------------------------------------------------------
            # dugum_verisi: CSV'den okuduğumuz dev tablonun tamamı (DataFrame).
            # .iterrows() : Bu tabloyu parçalar ve her seferinde bize BİR SATIR verir.
            #               Döngü her döndüğünde sıradaki satıra geçer.
            for indeks, satir in dugum_verisi.iterrows():
                # indeks : O an kaçıncı satırdayız? (0, 1, 2... diye sayar).
                #          (Genelde kullanmayız ama fonksiyon mecbur verir).

                # satir  : O anki satırın içindeki tüm bilgiler (node_id, s_ms, r_node).
                #          Bu bir paket gibidir, içinden istediğimiz sütunu seçeriz.

                # ---------------------------------------------------------------------
                # VERİYİ ÇEKME VE TEMİZLEME
                # ---------------------------------------------------------------------
                # 1. satir['node_id']: O satırdaki 'node_id' sütununda ne yazıyorsa onu getir.
                # 2. int(...)        : Gelen veriyi TAM SAYIYA çevir.
                # CSV dosyasında sayılar tam sayı (örn: 5) görünse bile,
                # Pandas bazen bunları ondalıklı (örn: 5.0) olarak okuyabilir.
                # int() diyerek "Bunu kesinlikle tam sayı yap" diyoruz.
                # Bu sayede ileride "5.0" ile "5" karışıklığı (Type Mismatch) çıkmasını engelliyoruz.
                dugum_id = int(satir['node_id'])

                # CSV'deki sütunları değişkenlere alıyoruz
                islem_suresi = float(satir['s_ms'])  # İşlem süresi (milisaniye)
                guvenilirlik = float(satir['r_node'])  # Düğüm güvenilirliği (0-1 arası)

                # Düğümü grafa ekle ve özelliklerini kaydet
                self.graf.add_node(dugum_id)
                self.graf.nodes[dugum_id]['islem_suresi'] = islem_suresi
                self.graf.nodes[dugum_id]['guvenilirlik'] = guvenilirlik

                # Matematiksel Dönüşüm: Güvenilirlik çarpımsal olduğu için (-log) alarak
                # toplamsal maliyete çeviriyoruz. Algoritma toplama yapabilsin diye.
                self.graf.nodes[dugum_id]['guv_maliyeti'] = -math.log(guvenilirlik)

            print(f"✅ {len(dugum_verisi)} adet düğüm başarıyla yüklendi.")

        except FileNotFoundError:
            print(f"❌ HATA: '{DUGUM_DOSYASI}' dosyası bulunamadı! Lütfen dosya ismini kontrol et.")
            return

        # ---------------------------------------------------------
        # ADIM 2: KENAR (LINK) VERİLERİNİ YÜKLEME
        # ---------------------------------------------------------
        try:
            kenar_verisi = pd.read_csv(KENAR_DOSYASI, sep=';', decimal=',')

            for indeks, satir in kenar_verisi.iterrows():
                kaynak_dugum = int(satir['src'])  # Nereden (Source)
                hedef_dugum = int(satir['dst'])  # Nereye (Destination)

                # Özellikleri alıyoruz
                kapasite = float(satir['capacity_mbps'])  # Bant genişliği
                gecikme = float(satir['delay_ms'])  # Kablo gecikmesi
                guvenilirlik = float(satir['r_link'])  # Hat güvenilirliği

                # Bağlantıyı (Edge) grafa ekle
                self.graf.add_edge(kaynak_dugum, hedef_dugum)

                # Özellikleri kenara kaydet
                self.graf[kaynak_dugum][hedef_dugum]['bant_genisligi'] = kapasite
                self.graf[kaynak_dugum][hedef_dugum]['gecikme'] = gecikme
                self.graf[kaynak_dugum][hedef_dugum]['guvenilirlik'] = guvenilirlik

                # KAYNAK MALİYETİ HESABI (PDF Formülü)
                # Formül: 1000 / Bant Genişliği
                # Bant genişliği ne kadar yüksekse maliyet o kadar az olur.
                self.graf[kaynak_dugum][hedef_dugum]['kaynak_maliyeti'] = 1000.0 / kapasite

                # GÜVENİLİRLİK MALİYETİ HESABI
                self.graf[kaynak_dugum][hedef_dugum]['guv_maliyeti'] = -math.log(guvenilirlik)

            print(f"✅ {len(kenar_verisi)} adet bağlantı başarıyla yüklendi.")

        except FileNotFoundError:
            print(f"❌ HATA: '{KENAR_DOSYASI}' dosyası bulunamadı!")
            return

        # ---------------------------------------------------------
        # ADIM 3: AĞ KONTROLÜ
        # ---------------------------------------------------------
        if nx.is_connected(self.graf):
            print("Bilgi: Ağ sorunsuz oluşturuldu. Tüm düğümler birbirine erişebilir.")
        else:
            print("UYARI: Ağ kopuk! Bazı düğümler arasında yol yok.")

        return self.graf

    def talep_listesini_getir(self):
        """
        DemandData.csv dosyasındaki 'Kimden -> Kime ne kadar trafik lazım'
        bilgisini okur ve liste olarak verir.
        """
        try:
            talep_verisi = pd.read_csv(TALEP_DOSYASI, sep=';', decimal=',')
            talep_listesi = []

            for indeks, satir in talep_verisi.iterrows():
                # Her bir talep satırını sözlük (dictionary) yapıyoruz
                talep = {
                    'kaynak': int(satir['src']),
                    'hedef': int(satir['dst']),
                    'miktar': float(satir['demand_mbps'])
                }
                talep_listesi.append(talep)

            return talep_listesi

        except FileNotFoundError:
            print(f"❌ HATA: '{TALEP_DOSYASI}' dosyası bulunamadı.")
            return []

    def yol_maliyeti_hesapla(self, yol):
        """
        Verilen bir yolun (path) toplam maliyetini (Fitness) hesaplar.
        Parametre: yol -> [0, 5, 24] gibi düğüm listesi.
        """
        if not yol or len(yol) < 2:
            return float('inf'), 0, 0, 0

        toplam_gecikme = 0
        toplam_guv_maliyeti = 0
        toplam_kaynak_maliyeti = 0

        # --- A. YOL ÜZERİNDEKİ KENARLARIN (LINK) MALİYETİ ---
        for i in range(len(yol) - 1):
            u = yol[i]
            v = yol[i + 1]
            baglanti = self.graf[u][v]  # O iki düğüm arasındaki hat bilgisi

            toplam_gecikme += baglanti['gecikme']
            toplam_guv_maliyeti += baglanti['guv_maliyeti']
            toplam_kaynak_maliyeti += baglanti['kaynak_maliyeti']

        # --- B. DÜĞÜMLERİN (NODE) KENDİ MALİYETLERİ ---
        # 1. Güvenilirlik: Yol üzerindeki HER düğüm risk oluşturur, hepsi toplanır.
        for dugum_id in yol:
            toplam_guv_maliyeti += self.graf.nodes[dugum_id]['guv_maliyeti']

        # 2. İşlem Gecikmesi: Sadece ARADAKİ düğümlerde vakit kaybedilir.
        # Kaynak (ilk) ve Hedef (son) düğümde işlem gecikmesi sayılmaz (PDF kuralı).
        ara_dugumler = yol[1:-1]
        for dugum_id in ara_dugumler:
            toplam_gecikme += self.graf.nodes[dugum_id]['islem_suresi']

        # --- C. GENEL SKOR (FITNESS) HESABI ---
        # Ağırlıklı toplama yöntemi ile tek bir skor üretiyoruz.
        genel_skor = (self.w_gecikme * toplam_gecikme) + \
                     (self.w_guvenilirlik * toplam_guv_maliyeti) + \
                     (self.w_kaynak * toplam_kaynak_maliyeti)

        return genel_skor, toplam_gecikme, toplam_guv_maliyeti, toplam_kaynak_maliyeti


# =============================================================================
# ÇALIŞTIRMA KISMI (MAIN)
# =============================================================================
if __name__ == "__main__":
    print("\n--- PROJE BAŞLATILIYOR (VERİ OKUMA MODU) ---")

    # 1. Sınıfı oluştur
    ag_yoneticisi = AgOrtami()

    # 2. Dosyaları oku ve ağı kur
    ag_yoneticisi.verileri_yukle_ve_agi_kur()

    # 3. Talepleri oku
    talepler = ag_yoneticisi.talep_listesini_getir()

    if len(talepler) > 0:
        # Örnek olarak listedeki İLK talebi alıp test edelim
        ornek_talep = talepler[0]
        src = ornek_talep['kaynak']
        dst = ornek_talep['hedef']
        bw = ornek_talep['miktar']

        print(f"\n🧪 TEST: Talep Dosyasından İlk Kayıt Deneniyor...")
        print(f"   Kaynak: {src} -> Hedef: {dst} | İstenen Hız: {bw} Mbps")

        try:
            # NetworkX kütüphanesinin hazır fonksiyonu ile en kısa yolu buluyoruz.
            # (Bu sadece test amaçlıdır, projenin ilerleyen kısmında burayı senin algoritman yapacak)
            bulunan_yol = nx.shortest_path(ag_yoneticisi.graf, src, dst)
            print(f"   📍 Bulunan Yol: {bulunan_yol}")

            # Bu yolun maliyetini hesaplatıyoruz
            skor, gecikme, guv_maliyet, kaynak_maliyet = ag_yoneticisi.yol_maliyeti_hesapla(bulunan_yol)

            # Sonuçları ekrana yaz
            # Güvenilirliği tekrar yüzdeye çeviriyoruz (e üzeri -maliyet)
            guvenilirlik_yuzde = math.exp(-guv_maliyet) * 100

            print("\n📊 HESAPLANAN DEĞERLER:")
            print(f"   1. Toplam Gecikme    : {gecikme:.2f} ms")
            print(f"   2. Toplam Güvenilirlik: %{guvenilirlik_yuzde:.4f}")
            print(f"   3. Kaynak Maliyeti   : {kaynak_maliyet:.2f}")
            print(f"   --------------------------------------")
            print(f"   ★ GENEL SKOR (Fitness): {skor:.4f} (Daha düşük daha iyi)")

        except nx.NetworkXNoPath:
            print("   ⚠️ Hata: Bu iki düğüm arasında fiziksel bir yol yok!")
    else:
        print("   ⚠️ Talep listesi boş geldi, dosya içeriğini kontrol edin.")