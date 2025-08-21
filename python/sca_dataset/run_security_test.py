import sca_dataset as d
import numpy as np
from tqdm import tqdm, trange
from random import randint, choice
import matplotlib.pyplot as plt

import time

import config # type: ignore

nbs = config.nbs

def compute_one(nbt_a, i):
    y = d.Dataset("Hamming_Weight", ["delay"], ["Boolean, Software", "1"], nbt_a, nbs, 1, False)
    
    plaintext2 = np.zeros((nbt_a, y.nb_bytes), dtype=int)
    for l in range(nbt_a):
        for j in range(y.nb_bytes):
            plaintext2[l][j] = randint(0,255)

    trace2, label2, s2, mask = y.traces(plaintext2, 30.0, [], 5*i + 1, y.mask(), [])
    rang, nbt = d.soge(trace2, plaintext2[:, 0], 1, s2[0], 100)

    idx = len(rang) - 1
    if 0 <= idx < len(nbt):
        return nbt[idx]
    else:
        return 0

def compute_one_wrapper(args):
    return compute_one(*args)

def worker_i(i, nbt_a, nb_mean):
    from multiprocessing import Pool
    from tqdm import tqdm

    args = [(nbt_a, i) for k in range(nb_mean)]
    with Pool() as pool:
        results = list(tqdm(pool.imap_unordered(compute_one_wrapper, args), total=nb_mean))
    return np.mean(results)

def run_security_test_multiprocessing(nbt_a, nb_mean, nb_bytes):
    results = []
    for i in trange(nb_bytes):
        res = worker_i(i, nbt_a, nb_mean)
        results.append(res)

    res_array = np.array(results)

    np.save(f'./resultat/impact_shuffling_{time.time()}', res_array)

    plt.figure(figsize=(10, 6))

    plt.plot(res_array, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=6, label='Données')

    # plt.plot(x, linestyle='-', color="#aa2f2f", linewidth=2, markersize=6, 

    plt.title("Impact du Shuffling-Masquage Ordre 1 sur une CPA selon le nombre d'bytes", fontsize=16, fontweight='bold')
    plt.xlabel("Nombre d'bytes", fontsize=14)
    plt.ylabel("Nombre de traces pour retrouver la clé", fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.legend(fontsize=12)
    plt.tight_layout()

    plt.savefig(f"./resultat/graph_shuffling_{time.time()}.pdf")
