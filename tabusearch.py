# ============================================================
# TABU SEARCH ALGORİTMASI
# network_model.py ÜZERİNDEN ÇALIŞIR
# BAŞLANGIÇ YOLU: nx.shortest_path (DÜZELTİLDİ)
# ============================================================
import networkx as nx

def komsu_yollar_uret(graf, mevcut_yol):
    komsular = []
    if len(mevcut_yol) < 3:
        return komsular
    for i in range(1, len(mevcut_yol) - 1):
        onceki = mevcut_yol[i - 1]
        sonraki = mevcut_yol[i + 1]
        # Ortak komşuları bul (yolu değiştirmek için)
        alternatifler = set(graf.neighbors(onceki)).intersection(graf.neighbors(sonraki))
        for dugum in alternatifler:
            if dugum not in mevcut_yol:
                yeni_yol = mevcut_yol[:i] + [dugum] + mevcut_yol[i + 1:]
                komsular.append(yeni_yol)
    return komsular

def tabu_search(ag_objesi, kaynak, hedef, istenen_bw, iterasyon=50, tabu_boyutu=10):
    graf = ag_objesi.graf
    try:
        mevcut_yol = nx.shortest_path(graf, kaynak, hedef)
    except nx.NetworkXNoPath:
        return None

    en_iyi_yol = mevcut_yol
    # Senin network_model.py içindeki fonksiyonu kullanıyoruz
    en_iyi_skor, _, _, _ = ag_objesi.yol_maliyeti_hesapla(mevcut_yol, istenen_bw)
    tabu_listesi = []

    for _ in range(iterasyon):
        komsu_yollar = komsu_yollar_uret(graf, mevcut_yol)
        en_iyi_komsu = None
        en_iyi_komsu_skor = float('inf')

        for yol in komsu_yollar:
            if yol in tabu_listesi: continue
            
            # Maliyet hesabı
            skor, _, _, _ = ag_objesi.yol_maliyeti_hesapla(yol, istenen_bw)
            
            if skor < en_iyi_komsu_skor:
                en_iyi_komsu = yol
                en_iyi_komsu_skor = skor

        if en_iyi_komsu is None: break
        
        mevcut_yol = en_iyi_komsu
        tabu_listesi.append(mevcut_yol)
        if len(tabu_listesi) > tabu_boyutu: tabu_listesi.pop(0)

        if en_iyi_komsu_skor < en_iyi_skor:
            en_iyi_yol = en_iyi_komsu
            en_iyi_skor = en_iyi_komsu_skor

    return en_iyi_yol
