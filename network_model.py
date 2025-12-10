# 1. NETWORKX (Ağ Grafiği Kütüphanesi)
# Neden Gerekli?: Projenin temeli olan "Düğüm" (Node) ve "Bağlantı" (Link) yapısını kurmak için.
# Projedeki Görevi:
# - PDF [25]: Erdos-Renyi G(n,p) modeline göre 250 düğümlü karmaşık ağın oluşturulmasını sağlar.
# - PDF [26]: Ağın "bağlı" (connected) olup olmadığını kontrol eder (nx.is_connected).
# - Düğümler ve kenarlar üzerindeki verileri (attribute) saklar.
import networkx as nx

# 2. RANDOM (Rastgele Sayı Üreteci)
# Neden Gerekli?: Proje "Deterministik" (sabit) değil, "Stokastik" (rastgele) değerler üzerine kurulu.
# Projedeki Görevi:
# - PDF [22]: Ağ topolojisinin her çalıştırıldığında rastgele oluşturulmasını sağlar.
# - PDF [29, 30, 33, 34]: Bant genişliği, gecikme ve güvenilirlik değerlerini
#   belirtilen aralıklarda (örn: 3ms - 15ms) rastgele atar.
# - PDF [116]: "Seed" (tohum) mekanizması ile deneylerin tekrarlanabilir olmasını sağlar (random.seed).
import random

# 3. MATH (Matematik Fonksiyonları)
# Neden Gerekli?: Özellikle "Güvenilirlik" hesabındaki özel bir dönüşüm işlemi için.
# Projedeki Görevi:
# - PDF [48-52]: Güvenilirlik normalde çarpımsal (multiplicative) bir metriktir (0.99 * 0.98...).
# - PDF [52]: Ancak optimizasyon algoritmaları "toplama" ile daha iyi çalışır.
# - Math kütüphanesi, "-math.log(guvenilirlik)" fonksiyonu ile bu çarpma işlemini
#   toplama işlemine (Cost) dönüştürmek için kullanılır.
import math
# =============================================================================
DUGUM_SAYISI = 250  # N: Toplam düğüm sayısı
BAGLANTI_OLASILIGI = 0.4  # P: Bağlantı olasılığı (Erdos-Renyi için)


class AgOrtami:
    """
    Bu sınıf, projenin fiziksel ağ altyapısını oluşturur, yönetir ve
    yol maliyetlerini hesaplar.

    Özellikler:
    - Erdos-Renyi modeli ile topoloji oluşturma.
    - Seed (tohum) desteği ile tekrarlanabilir deneyler.
    - PDF'teki kısıtlara göre bant genişliği, gecikme ve güvenilirlik atamaları.
    """

    def __init__(self, seed=42):
        """
        Başlatıcı metod.

        Parametreler:
            seed (int): Rastgele sayı üretecinin başlangıç değeri.
                        Aynı seed değeri her zaman aynı ağ yapısını üretir.
                        (PDF 'Teslim Edilecekler' maddesi gereği gereklidir.)
        """
        self.seed = seed
        self.graf = nx.Graph()

        # Rastgeleliği sabitle (Reprodüklibilite için)
        random.seed(self.seed)

        # PDF [66-68]: Varsayılan Optimiazsyon Ağırlıkları
        # Bu değerler kullanıcı arayüzünden (GUI) dinamik olarak değiştirilebilir.
        self.w_gecikme = 0.33  # Gecikme ağırlığı
        self.w_guvenilirlik = 0.33  # Güvenilirlik ağırlığı
        self.w_kaynak = 0.34  # Kaynak kullanım ağırlığı

    def topoloji_olustur(self):
        """
        PDF [22, 26]: Rastgele ve 'bağlı' (connected) bir ağ oluşturur.
        Erdos-Renyi G(n,p) modeli kullanılır.
        """
        print(f"Topoloji oluşturuluyor... (Seed: {self.seed}, N={DUGUM_SAYISI}, P={BAGLANTI_OLASILIGI})")

        while True:
            # NetworkX'in Erdos-Renyi fonksiyonuna da seed veriyoruz.
            # Not: P=0.4 çok yüksek bir olasılık olduğu için ağın kopuk olma ihtimali
            # matematiksel olarak neredeyse sıfırdır, ama kural gereği kontrol ediyoruz.
            self.graf = nx.erdos_renyi_graph(n=DUGUM_SAYISI, p=BAGLANTI_OLASILIGI, seed=self.seed)

            # PDF Şartı: "Grafiğin bağlı (connected) olduğundan emin olunmalı"
            if nx.is_connected(self.graf):
                break
            else:
                print("Uyarı: Oluşan graf bağlı değil (kopuk). Seed değiştirilip tekrar deneniyor...")
                self.seed += 1  # Sonsuz döngüye girmemek için seed'i değiştir

        # Topoloji iskeleti oluştuktan sonra, fiziksel özellikleri (Delay, BW vb.) ata
        self._dugum_ozelliklerini_ata()
        self._baglanti_ozelliklerini_ata()

        print("✅ Ağ başarıyla oluşturuldu ve tüm fiziksel değerler atandı.")
        return self.graf

    def _dugum_ozelliklerini_ata(self):
        """
        PDF [27-30]: Her düğüme işlem süresi ve güvenilirlik değerlerini atar.
        """
        for dugum in self.graf.nodes():
            # 1. İşlem Süresi (Processing Delay)
            # PDF [29]: [0.5 ms - 2.0 ms] arasında rastgele bir değer.
            islem_suresi = random.uniform(0.5, 2.0)

            # 2. Düğüm Güvenilirliği (Node Reliability)
            # PDF [30]: [0.95 - 0.999] arasında rastgele bir değer.
            guvenilirlik = random.uniform(0.95, 0.999)

            # Değerleri düğüm nesnesine kaydet
            self.graf.nodes[dugum]['islem_suresi'] = islem_suresi
            self.graf.nodes[dugum]['guvenilirlik'] = guvenilirlik

            # PDF [52]: Minimizasyon problemi için Güvenilirlik Maliyeti (-log dönüşümü)
            # Çarpımsal metrikleri toplamsal hale getirmek için -ln(x) kullanılır.
            self.graf.nodes[dugum]['guv_maliyeti'] = -math.log(guvenilirlik)

    def _baglanti_ozelliklerini_ata(self):
        """
        PDF [31-35]: Her bağlantıya bant genişliği, gecikme ve güvenilirlik atar.
        """
        for u, v in self.graf.edges():
            # 1. Bant Genişliği (Bandwidth)
            # PDF [33]: [100 Mbps - 1000 Mbps] arasında rastgele.
            bant_genisligi = random.uniform(100, 1000)

            # 2. Bağlantı Gecikmesi (Link Delay)
            # PDF [34]: [3 ms - 15 ms] arasında rastgele.
            link_gecikmesi = random.uniform(3, 15)

            # 3. Bağlantı Güvenilirliği (Link Reliability)
            # PDF [35]: [0.95 - 0.999] arasında rastgele.
            link_guvenilirligi = random.uniform(0.95, 0.999)

            # Değerleri kenar (edge) nesnesine kaydet
            self.graf[u][v]['bant_genisligi'] = bant_genisligi
            self.graf[u][v]['gecikme'] = link_gecikmesi
            self.graf[u][v]['guvenilirlik'] = link_guvenilirligi

            # PDF [57]: Kaynak Maliyeti (Minimizasyon için)
            # Formül: ResourceCost = 1 Gbps / Bandwidth
            # 1 Gbps = 1000 Mbps olduğundan: 1000 / bant_genisligi
            self.graf[u][v]['kaynak_maliyeti'] = 1000.0 / bant_genisligi

            # PDF [52]: Güvenilirlik Maliyeti (-log dönüşümü)
            self.graf[u][v]['guv_maliyeti'] = -math.log(link_guvenilirligi)

    def yol_maliyeti_hesapla(self, yol):
        """
        PDF [38-68]: Verilen bir yolun (path) toplam maliyetini hesaplar.

        Parametreler:
            yol (list): Düğüm ID'lerinden oluşan liste. Örn: [0, 5, 24, 249]

        Dönüş:
            (toplam_skor, toplam_gecikme, toplam_guvenilirlik_maliyeti, toplam_kaynak_maliyeti)
        """
        # Geçersiz veya çok kısa yol kontrolü
        if not yol or len(yol) < 2:
            return float('inf'), 0, 0, 0

        toplam_gecikme = 0
        toplam_guv_maliyeti = 0
        toplam_kaynak_maliyeti = 0

        # --- A. KENAR (LINK) MALİYETLERİ ---
        # PDF [42, 52, 57]: Yol üzerindeki tüm bağlantıların değerleri toplanır.
        for i in range(len(yol) - 1):
            u, v = yol[i], yol[i + 1]
            kenar_verisi = self.graf[u][v]

            toplam_gecikme += kenar_verisi['gecikme']
            toplam_guv_maliyeti += kenar_verisi['guv_maliyeti']
            toplam_kaynak_maliyeti += kenar_verisi['kaynak_maliyeti']

        # --- B. DÜĞÜM (NODE) MALİYETLERİ ---
        kaynak_dugum = yol[0]  # Source (S)
        hedef_dugum = yol[-1]  # Destination (D)
        ara_dugumler = yol[1:-1]  # Yoldaki diğer düğümler

        # 1. Güvenilirlik: PDF'e göre S ve D dahil TÜM düğümlerin güvenilirliği hesaba katılır.
        toplam_guv_maliyeti += self.graf.nodes[kaynak_dugum]['guv_maliyeti']
        toplam_guv_maliyeti += self.graf.nodes[hedef_dugum]['guv_maliyeti']

        # 2. İşlem Gecikmesi: PDF [45, 49]
        # ÖNEMLİ KURAL: "Kaynak S ve Hedef D hariç tüm işlem gecikmelerinin toplamı"
        for dugum in ara_dugumler:
            dugum_verisi = self.graf.nodes[dugum]

            toplam_gecikme += dugum_verisi['islem_suresi']  # Sadece arada ekle
            toplam_guv_maliyeti += dugum_verisi['guv_maliyeti']  # Herkes için ekle

        # --- C. TOPLAM SKOR (FITNESS FUNCTION) ---
        # PDF [66]: Weighted Sum Method (Ağırlıklı Toplam Yöntemi)
        # TotalCost = (W_delay * Delay) + (W_rel * RelCost) + (W_res * ResCost)
        toplam_skor = (self.w_gecikme * toplam_gecikme) + \
                      (self.w_guvenilirlik * toplam_guv_maliyeti) + \
                      (self.w_kaynak * toplam_kaynak_maliyeti)

        return toplam_skor, toplam_gecikme, toplam_guv_maliyeti, toplam_kaynak_maliyeti


# =============================================================================
# MODÜL TEST ALANI
# Bu dosya doğrudan çalıştırıldığında (import edilmediğinde) burası çalışır.
# =============================================================================
if __name__ == "__main__":
    print("\n--- AĞ OLUŞTURUCU TESTİ BAŞLIYOR ---")

    # 1. Ortamı Başlat (Seed=42 ile)
    # Seed verdiğimiz için her çalıştırmada AYNI sonuçları görmeliyiz.
    ortam = AgOrtami(seed=42)
    ortam.topoloji_olustur()

    # 2. Rastgele bir Kaynak ve Hedef seç (Test amaçlı)
    tum_dugumler = list(ortam.graf.nodes())
    S = tum_dugumler[0]
    D = tum_dugumler[-1]

    print(f"\nTest Rotası: {S} -> {D}")

    # 3. NetworkX'in kendi en kısa yol algoritması ile bir yol bulalım
    # Not: Bu, bizim maliyet fonksiyonumuzu değil, sadece adım sayısını baz alır.
    try:
        ornek_yol = nx.shortest_path(ortam.graf, S, D)
        print(f"NetworkX Shortest Path: {ornek_yol}")

        # 4. Bulunan yolun maliyetlerini hesaplayalım.
        skor, gecikme, guv, kaynak = ortam.yol_maliyeti_hesapla(ornek_yol)

        print("\n--- HESAPLANAN METRİKLER ---")
        print(f"1. Toplam Gecikme (Delay): {gecikme:.2f} ms")
        # Matematiksel ters işlem yapıyoruz: e üzeri (-maliyet)
        gercek_oran = math.exp(-guv) * 100
        # Hem yüzdeyi hem de parantez içinde maliyeti yazdırıyoruz
        print(f"2. Güvenilirlik: %{gercek_oran:.2f} (Maliyet: {guv:.4f})")
        print(f"3. Kaynak Kullanımı (Maliyet): {kaynak:.2f}")
        print(f"--------------------------------------")
        print(f"★ GENEL SKOR (Fitness): {skor:.4f}")

    except nx.NetworkXNoPath:
        print("Hata: Bu iki düğüm arasında yol bulunamadı!")
        # Deneme 2

# ============================================================
#   GENETIC ALGORITHM OPERATÖRLERİ
# ============================================================


# RANDOM PATH (hızlı yol oluşturucu)
def random_valid_path(graph, S, D, max_len=80):
    path = [S]
    visited = {S}
    current = S

    for _ in range(max_len):
        if current == D:
            return path

        neighbors = list(graph.neighbors(current))
        if not neighbors:
            return None

        nxt = random.choice(neighbors)
        if nxt in visited:
            continue

        visited.add(nxt)
        path.append(nxt)
        current = nxt

    return None


# ============================================================
# PATH REPAIR (geçersiz yolları düzeltir)
# ============================================================

def repair_path(graph, path, S, D):
    if path[0] != S:
        path = [S] + path
    if path[-1] != D:
        path = path + [D]

    repaired = [path[0]]

    for i in range(len(path) - 1):
        u = repaired[-1]
        v = path[i + 1]

        if v in graph.neighbors(u):
            repaired.append(v)
        else:
            try:
                fix = nx.shortest_path(graph, u, v)
                repaired.extend(fix[1:])
            except:
                alt = random_valid_path(graph, u, D)
                repaired.extend(alt[1:])
                return repaired

    return repaired


# ============================================================
# GA OPERATORS
# ============================================================

def tournament_selection(pop, fitness, k=5):
    group = random.sample(pop, k)
    return min(group, key=lambda p: fitness[tuple(p)])


def crossover(p1, p2):
    if len(p1) < 3 or len(p2) < 3:
        return p1[:], p2[:]

    c1 = random.randint(1, len(p1) - 2)
    c2 = random.randint(1, len(p2) - 2)

    child1 = p1[:c1] + p2[c2:]
    child2 = p2[:c2] + p1[c1:]

    return child1, child2


def mutate(path, rate=0.10):
    if random.random() < rate and len(path) > 3:
        i, j = random.sample(range(1, len(path) - 1), 2)
        path[i], path[j] = path[j], path[i]
    return path


# ============================================================
# FINAL GENETIC ALGORITHM (KISA SÜRÜM)
# ============================================================

def genetic_algorithm(ortam, S, D, pop_size=40, generations=120):
    graph = ortam.graf
    population = []

    # Başlangıç popülasyonu
    while len(population) < pop_size:
        p = random_valid_path(graph, S, D)
        if p:
            population.append(p)

    # Başlangıç fitness
    fitness = {
        tuple(p): ortam.yol_maliyeti_hesapla(p)[0]
        for p in population
    }

    best_path = None
    best_cost = float("inf")

    # Evrim döngüsü
    for gen in range(generations):

        new_pop = []

        # Elitism
        elite_n = max(1, pop_size // 10)
        elites = sorted(population, key=lambda p: fitness[tuple(p)])[:elite_n]
        new_pop.extend(elites)

        # Yeni nesil üret
        while len(new_pop) < pop_size:

            p1 = tournament_selection(population, fitness)
            p2 = tournament_selection(population, fitness)

            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)

            c1 = repair_path(graph, c1, S, D)
            c2 = repair_path(graph, c2, S, D)

            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        population = new_pop

        # Fitness güncelle
        fitness = {
            tuple(p): ortam.yol_maliyeti_hesapla(p)[0]
            for p in population
        }

        # En iyi birey
        best_gene = min(fitness, key=fitness.get)
        cost = fitness[best_gene]

        if cost < best_cost:
            best_cost = cost
            best_path = list(best_gene)

    return best_path, best_cost

