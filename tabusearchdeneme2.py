# ============================================================
# TABU SEARCH ALGORİTMASI
# network_model.py ÜZERİNDEN ÇALIŞIR
# ============================================================

import random
import networkx as nx
from network_model import AgOrtami


def rastgele_yol_uret(graf, kaynak, hedef, max_adim=50):
    """
    Kaynak ile hedef arasında rastgele (ama geçerli) bir yol üretir.
    """
    yol = [kaynak]
    mevcut = kaynak

    for _ in range(max_adim):
        komsular = list(graf.neighbors(mevcut))
        if not komsular:
            break

        secilen = random.choice(komsular)

        if secilen in yol:
            continue

        yol.append(secilen)
        mevcut = secilen

        if mevcut == hedef:
            return yol

    return None


def komsu_yollar_uret(graf, mevcut_yol):
    """
    Mevcut yolun içinden bir düğümü değiştirerek komşu çözümler üretir.
    """
    komsular = []

    if len(mevcut_yol) < 3:
        return komsular

    for i in range(1, len(mevcut_yol) - 1):
        onceki = mevcut_yol[i - 1]
        sonraki = mevcut_yol[i + 1]

        alternatifler = set(graf.neighbors(onceki)).intersection(
            graf.neighbors(sonraki)
        )

        for dugum in alternatifler:
            if dugum not in mevcut_yol:
                yeni_yol = mevcut_yol[:i] + [dugum] + mevcut_yol[i + 1:]
                komsular.append(yeni_yol)

    return komsular


def tabu_search(ag, kaynak, hedef, istenen_bw,
                iterasyon_sayisi=100, tabu_listesi_boyutu=20):
    """
    Tabu Search algoritması ana fonksiyonu
    """

    graf = ag.graf

    # -------------------------------
    # BAŞLANGIÇ ÇÖZÜMÜ
    # -------------------------------
    mevcut_yol = rastgele_yol_uret(graf, kaynak, hedef)

    if mevcut_yol is None:
        print("❌ Başlangıç yolu bulunamadı.")
        return None, float('inf')

    en_iyi_yol = mevcut_yol
    en_iyi_skor, _, _, _ = ag.yol_maliyeti_hesapla(
        mevcut_yol, istenen_bw
    )

    tabu_listesi = []

    # -------------------------------
    # ANA DÖNGÜ
    # -------------------------------
    for iterasyon in range(iterasyon_sayisi):

        komsu_yollar = komsu_yollar_uret(graf, mevcut_yol)
        en_iyi_komsu = None
        en_iyi_komsu_skor = float('inf')

        for yol in komsu_yollar:
            if yol in tabu_listesi:
                continue

            skor, _, _, _ = ag.yol_maliyeti_hesapla(yol, istenen_bw)

            if skor < en_iyi_komsu_skor:
                en_iyi_komsu = yol
                en_iyi_komsu_skor = skor

        if en_iyi_komsu is None:
            break

        mevcut_yol = en_iyi_komsu

        # TABU GÜNCELLE
        tabu_listesi.append(mevcut_yol)
        if len(tabu_listesi) > tabu_listesi_boyutu:
            tabu_listesi.pop(0)

        # GLOBAL EN İYİ GÜNCELLE
        if en_iyi_komsu_skor < en_iyi_skor:
            en_iyi_yol = en_iyi_komsu
            en_iyi_skor = en_iyi_komsu_skor

        print(f"Iterasyon {iterasyon + 1} | En iyi skor: {en_iyi_skor:.4f}")

    return en_iyi_yol, en_iyi_skor


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n--- TABU SEARCH BAŞLATILIYOR ---")

    # AĞI OLUŞTUR
    ag_ortami = AgOrtami()
    ag_ortami.verileri_yukle_ve_agi_kur()

    # TALEPLERİ AL
    talepler = ag_ortami.talep_listesini_getir()
    talep = talepler[0]

    kaynak = talep['kaynak']
    hedef = talep['hedef']
    istenen_bw = talep['miktar']

    print(f"Kaynak: {kaynak}, Hedef: {hedef}, BW: {istenen_bw}")

    # TABU SEARCH ÇALIŞTIR
    en_iyi_yol, en_iyi_skor = tabu_search(
        ag_ortami,
        kaynak,
        hedef,
        istenen_bw
    )

    print("\n--- SONUÇ ---")
    print(f"En iyi yol: {en_iyi_yol}")
    print(f"Toplam Fitness (Maliyet): {en_iyi_skor:.4f}")
