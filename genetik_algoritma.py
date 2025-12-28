import random
import networkx as nx
from network_model import AgOrtami

# RANDOM PATH Başlangıç popülasyonu

def random_valid_path(graph, S, D, max_len=80):
    path = [S]
    visited = {S}
    current = S

    for _ in range(max_len):
        if current == D:
            return path

        neighbors = list(graph.neighbors(current))
        candidates = [n for n in neighbors if n not in visited]
        if not candidates:
            return None

        nxt = random.choice(candidates)
        visited.add(nxt)
        path.append(nxt)
        current = nxt

    return None



# PATH REPAIR

def repair_path(graph, path, S, D):
    if not path:
        return None

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
                return None

    return repaired



# SELECTION – TOURNAMENT

def tournament_selection(population, fitness, k=5):
    k = min(k, len(population))
    group = random.sample(population, k)
    return min(group, key=lambda p: fitness[tuple(p)])



# ELITISM

def elitism(population, fitness, rate=0.1):
    elite_n = max(1, int(len(population) * rate))
    sorted_pop = sorted(population, key=lambda p: fitness[tuple(p)])
    return [p.copy() for p in sorted_pop[:elite_n]]



# CROSSOVER

def crossover(p1, p2):
    if len(p1) < 3 or len(p2) < 3:
        return p1[:], p2[:]

    c1 = random.randint(1, len(p1) - 2)
    c2 = random.randint(1, len(p2) - 2)

    return p1[:c1] + p2[c2:], p2[:c2] + p1[c1:]



# MUTATION

def mutate(path, rate=0.1):
    if random.random() < rate and len(path) > 3:
        i, j = random.sample(range(1, len(path) - 1), 2)
        path[i], path[j] = path[j], path[i]
    return path


# GENETIC ALGORITHM (ANA)

def genetic_algorithm(ortam, S, D, bw, pop_size=50, generations=100):
    graph = ortam.graf
    population = []

    #Başlangıç Popülasyonu
    attempts = 0
    while len(population) < pop_size and attempts < pop_size * 10:
        p = random_valid_path(graph, S, D)
        if p:
            population.append(p)
        attempts += 1

    if not population:
        return None, float('inf')

    #Fitness
    fitness = {
        tuple(p): ortam.yol_maliyeti_hesapla(p, istenen_bw=bw)[0]
        for p in population
    }

    best_path = None
    best_cost = float('inf')

    # Evrim Döngüsü
    for _ in range(generations):
        new_population = []

        # Elitism
        elites = elitism(population, fitness)
        new_population.extend(elites)

        elite_cost = fitness[tuple(elites[0])]
        if elite_cost < best_cost:
            best_cost = elite_cost
            best_path = elites[0].copy()

        # Yeni bireyler
        while len(new_population) < pop_size:
            p1 = tournament_selection(population, fitness)
            p2 = tournament_selection(population, fitness)

            c1, c2 = crossover(p1, p2)
            c1 = repair_path(graph, mutate(c1), S, D)
            c2 = repair_path(graph, mutate(c2), S, D)

            if c1:
                new_population.append(c1)
            if len(new_population) < pop_size and c2:
                new_population.append(c2)

        population = new_population


        fitness = {
            tuple(p): ortam.yol_maliyeti_hesapla(p, istenen_bw=bw)[0]
            for p in population
        }

    return best_path, best_cost


if __name__ == "__main__":


    ortam = AgOrtami()
    ortam.verileri_yukle_ve_agi_kur()


    talepler = ortam.talep_listesini_getir()
    talep = talepler[0]

    S = talep["kaynak"]
    D = talep["hedef"]
    bw = talep["miktar"]


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
        print("✖ Uygun bir yol bulunamadı.")

