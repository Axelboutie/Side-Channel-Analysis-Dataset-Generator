#!/usr/bin/env python3
import logging

import sca_dataset as d
import numpy as np
import matplotlib.pyplot as plt
import h5py as h
import time

from run_security_test import run_security_test_multiprocessing # type: ignore
from batch import batch # type: ignore
import config # type: ignore

FORMAT = '%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT)
logging.getLogger().setLevel(logging.INFO)




def main():

    nbt_a = config.nbt_a
    nbs = config.nbs
    nb_bytes = config.nb_bytes
    nbt_p = config.nbt_p
    nb_mean = config.nb_mean

    y = d.Dataset("Hamming_Distance", ["delay"], ["Boolean, Hardware", "1"], nbt_a, nbs, nb_bytes,True)
    print(y)

    y.gen_file(5.0, [], 5, "./resultat/test.h5", nbt_p, nbt_a, 0.50, False)

    # poi = d.gest_poi(nbs, nb_bytes)
    # print(poi)
    # t = y.t_ref(poi,0.5)
    # print(len(t))

    # trace, labels, key, maks = y.traces(plaintext1, 100, [], 2, y.mask() ,t)

    f = h.File("./resultat/Delay.h5", "r")

    # trace = np.array(f["Profiling_traces/traces"])
    trace2 = np.array(f["Attack_traces/traces"])
    label= np.array(f["Attack_traces/Metadata/plaintext"])

  
    # # maks = np.array(f["Attack_traces/Metadata/masks"])

    # # print(trace_p)
    # # print(maks)
    # # test_shuffle_multiprocessing(nbt_a, nb_moy, nb_bytes)

    trace = np.array(trace2)
    labels = np.array(label)
    # plt.plot(trace[4])

    key, correl = d.cpa_hw(trace, labels[:,0])
    print(key)

    
    # # plt.plot(trace[199])
    # # plt.plot(trace2[5000])
    # # plt.plot(trace[20])
    # plt.show()

    print("C'est bon")
main()


