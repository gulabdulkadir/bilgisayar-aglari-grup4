import random
import networkx as nx  # repair_path fonksiyonu için gerekli

from network_model import AgOrtami
import time  # Süre ölçümü için gerekli

# ============================================================
#   GENETIC ALGORITHM OPERATÖRLERİ
# ============================================================

# RANDOM PATH (Başlangıç popülasyonu üretici)
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

        # Ziyaret edilmemiş komşuları filtrele (Döngüye girmemek için)
        candidates = [n for n in neighbors if n not in visited]
        if not candidates:
            return None

        nxt = random.choice(candidates)
        visited.add(nxt)
        path.append(nxt)
        current = nxt

    return None


# ============================================================
# PATH REPAIR (Geçersiz/Kopuk yolları tamir eder)
# ============================================================
def repair_path(graph, path, S, D):
    # Başlangıç ve Bitiş noktalarını zorla
    if not path:
        path = [S]
    if path[0] != S:
        path = [S] + path
    if path[-1] != D:
        path = path + [D]

    repaired = [path[0]]

    for i in range(len(path) - 1):
        u = repaired[-1]
        v = path[i + 1]

        # Eğer u ve v doğrudan bağlıysa ekle
        if v in graph.neighbors(u):
            repaired.append(v)
        else:
            # Bağlı değilse araya en kısa yolu yama yap
            try:
                fix = nx.shortest_path(graph, u, v)
                repaired.extend(fix[1:])  # İlk elemanı atla (zaten var)
            except:
                # O da olmazsa rastgele tamamla
                alt = random_valid_path(graph, u, D)
                if alt:
                    repaired.extend(alt[1:])
                    return repaired
                else:
                    return None  # Tamir edilemedi

    return repaired


# ============================================================
# GA OPERATÖRLERİ (Selection, Crossover, Mutation)
# ============================================================

   # [cite: 33] Turnuva Seçimi: Rastgele bireylerden en iyisini seçer


def tournament_selection(pop, fitness, k=5):
    # Popülasyon küçükse k değerini düşür
    if len(pop) < k:
        k = len(pop)
    group = random.sample(pop, k)
    # Fitness değeri en DÜŞÜK olanı (en az maliyetliyi) seç
    return min(group, key=lambda p: fitness[tuple(p)])


  # Çaprazlama: İki ebeveynin genlerini karıştırır
def crossover(p1, p2):
    # Çok kısa yollarda çaprazlama yapma
    if len(p1) < 3 or len(p2) < 3:
        return p1[:], p2[:]

    # Kesim noktalarını belirle
    try:
        c1 = random.randint(1, len(p1) - 2)
        c2 = random.randint(1, len(p2) - 2)
    except ValueError:
        return p1[:], p2[:]

    # Parçaları birleştir
    child1 = p1[:c1] + p2[c2:]
    child2 = p2[:c2] + p1[c1:]

    return child1, child2


  # [cite: 52] Mutasyon: Çeşitlilik için düğüm yer değiştirir


def mutate(path, rate=0.10):
    if random.random() < rate and len(path) > 3:
        try:
            i, j = random.sample(range(1, len(path) - 1), 2)
            path[i], path[j] = path[j], path[i]
        except ValueError:
            pass  # Yeterli uzunluk yoksa pas geç
    return path


# ============================================================
# GENETİK ALGORİTMA (ANA FONKSİYON)
# ============================================================



def genetic_algorithm(ortam, S, D, bw, pop_size=50, generations=100):
    graph = ortam.graf
    population = []

    start_time = time.time()  # Başlangıç zamanı
    print(f"\n🧬 GA Başlıyor: {S} -> {D} (Talep: {bw} Mbps)")
    print(f"   Popülasyon: {pop_size} | Nesil Sayısı: {generations}")
    print("-" * 60)
    print(f"{'Nesil':<10} | {'En İyi Maliyet':<20} | {'Ortalama Maliyet':<20}")
    print("-" * 60)

    # 1. BAŞLANGIÇ POPÜLASYONU
    attempts = 0
    while len(population) < pop_size and attempts < pop_size * 10:  # Deneme sayısını artırdık
        p = random_valid_path(graph, S, D)
        if p:
            population.append(p)
        attempts += 1

    if not population:
        print("❌ HATA: Başlangıç popülasyonu oluşturulamadı!")
        return None, float('inf')

    # Başlangıç Fitness Hesaplama
    fitness = {
        tuple(p): ortam.yol_maliyeti_hesapla(p, istenen_bw=bw)[0]
        for p in population
    }

    best_path = None
    best_cost = float("inf")

    # Gelişimi takip etmek için liste
    history = []

    # 2. EVRİM DÖNGÜSÜ
    for gen in range(1, generations + 1):  # 1'den başlatıyoruz görüntü güzel olsun diye
        new_pop = []

        # Elitism
        elite_n = max(1, int(pop_size * 0.1))
        sorted_pop = sorted(population, key=lambda p: fitness.get(tuple(p), float('inf')))
        elites = sorted_pop[:elite_n]
        new_pop.extend(elites)

        # En iyiyi kaydet
        current_best = elites[0]
        current_cost = fitness[tuple(current_best)]

        if current_cost < best_cost:
            best_cost = current_cost
            best_path = list(current_best)

        # Yeni bireyler üret
        while len(new_pop) < pop_size:
            p1 = tournament_selection(population, fitness)
            p2 = tournament_selection(population, fitness)
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            c1 = repair_path(graph, c1, S, D)
            c2 = repair_path(graph, c2, S, D)
            if c1: new_pop.append(c1)
            if len(new_pop) < pop_size and c2: new_pop.append(c2)

        population = new_pop

        # Fitness güncelle
        fitness = {}
        total_gen_cost = 0
        valid_count = 0

        for p in population:
            cost = ortam.yol_maliyeti_hesapla(p, istenen_bw=bw)[0]
            fitness[tuple(p)] = cost
            if cost != float('inf'):
                total_gen_cost += cost
                valid_count += 1

        # Ortalama maliyeti hesapla (Sadece geçerli yollar için)
        avg_cost = total_gen_cost / valid_count if valid_count > 0 else float('inf')
        history.append(best_cost)

        # HER 10 NESİLDE BİR BİLGİ VER
        if gen % 10 == 0 or gen == 1:
            print(f"{gen:<10} | {best_cost:<20.4f} | {avg_cost:<20.4f}")

    end_time = time.time()
    sure = end_time - start_time
    print("-" * 60)
    print(f"✅ GA Bitti. Süre: {sure:.2f} saniye. En iyi maliyet: {best_cost:.4f}")

    return best_path, best_cost


# ============================================================
# MAIN (TEST VE ÇALIŞTIRMA BLOĞU)
# ============================================================
if __name__ == "__main__":
    print("\n=== GENETİK ALGORİTMA BAŞLATILIYOR ===")
    try:
        # 1. Ağ Ortamını Hazırla
        ag_yoneticisi = AgOrtami()
        ag_yoneticisi.verileri_yukle_ve_agi_kur()

        # 2. Talep Listesini Çek
        talepler = ag_yoneticisi.talep_listesini_getir()

        if not talepler:
            print("⚠️ UYARI: Talep listesi boş veya okunamadı. DemandData.csv'yi kontrol et.")
        else:
            # Örnek olarak listedeki 1. talebi alıyoruz
            talep = talepler[0]

            kaynak = talep['kaynak']
            hedef = talep['hedef']
            bant_genisligi = talep['miktar']  # Bandwidth (Mbps)

            print(f"\n🧪 TEST SENARYOSU: Node {kaynak} -> Node {hedef}")
            print(f"   İstenen Kapasite: {bant_genisligi} Mbps")

            # 3. Genetik Algoritmayı Çalıştır
            best_path, best_cost = genetic_algorithm(
                ortam=ag_yoneticisi,
                S=kaynak,
                D=hedef,
                bw=bant_genisligi,  # Kapasite kontrolü için önemli!
                pop_size=50,  # Popülasyon boyutu (artırılabilir)
                generations=150  # Nesil sayısı (artırılabilir)
            )

            # 4. Sonuçları Ekrana Bas
            print("\n" + "=" * 40)
            if best_path:
                print(f"🏆 SONUÇ: En İyi Yol Bulundu!")
                print(f"📍 Rota: {best_path}")
                print(f"📉 Toplam Maliyet (Fitness): {best_cost:.4f}")

                # Detaylı maliyetleri görmek için tekrar hesaplatıyoruz
                _, d, r, rc = ag_yoneticisi.yol_maliyeti_hesapla(best_path, istenen_bw=bant_genisligi)
                print(f"   -----------------------------")
                print(f"   ⏱️  Toplam Gecikme : {d:.2f} ms")
                print(f"   🛡️  Güvenilirlik   : {r:.4f} (Maliyet)")
                print(f"   💰  Kaynak Maliyeti: {rc:.2f}")
            else:
                print("❌ SONUÇ: Uygun bir yol bulunamadı.")
                print("   Olası Sebepler:")
                print("   1. Ağ kopuk olabilir (fiziksel yol yok).")
                print("   2. Tüm yolların kapasitesi (bandwidth) yetersiz kalmış olabilir.")
            print("=" * 40 + "\n")

    except NameError as e:
        print(f"\n❌ HATA: {e}")
        print("💡 İPUCU: 'AgOrtami' sınıfı bu dosyada bulunamadı.")
        print("   Bu dosyayı çalıştırmadan önce AgOrtami sınıfını import ettiğinden emin ol.")
        print("   Örnek: 'from main import AgOrtami' satırını en başa ekle.")