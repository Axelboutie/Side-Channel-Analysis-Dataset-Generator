from tqdm import trange
import numpy as np
from random import randint
import sca_dataset as d


from gen_file import gen_file_att, gen_add # type: ignore

def batch(nbt_a, nbt_p, nbs, nb_bytes, data, batch_size, batch: bool, weights, delay):
    if batch == True:
        for i in range(2): #For the profiling group and the attack group
            if i == 0:
                data.nb_traces = nbt_a
                temp = False
                t = nbt_a
            else:
                data.nb_traces = nbt_p
                temp = True
                t = nbt_p
            if nbt_a % batch_size != 0 or nbt_p % batch_size != 0: #Check if the batch_sizes is working
                print("The batch_size must be a divisor of both your Attack's traces number and profiling number")
                raise TypeError
            else:
                data.nb_traces = batch_size
                if temp == True:
                    b = nbt_p / batch_size
                else:
                    b = nbt_a / batch_size
                # Loop to process all batches with a different case for the first iteration
                    plaintext1 = np.zeros((data.nb_traces, nb_bytes), dtype = int)
                    for l in range(data.nb_traces):
                        for j in range(nb_bytes):
                            plaintext1[l][j] = randint(0,255)

                    s1 = np.zeros(nb_bytes, dtype = int)

                    for le in range(nb_bytes):
                        s1[le] = randint(0,255)    

                    trace1, label1, secret= data.traces(plaintext1, 5.0, s1, weights, delay, data.mask(), data.t_ref(d.gen_poi(nbs, nb_bytes),0.50))
                    if i == 0:
                        gen_file_att(np.array(trace1, dtype=np.float16), np.array(label1, dtype=np.uint8),s1, plaintext1, t, nbs, temp)
                    else:
                        gen_add(np.array(trace1, dtype=np.float16), np.array(label1, dtype=np.uint8),s1, plaintext1, temp)
                
    else: #If we don't want to batch (It is preferable to use the gen_file function in rust)
        for i in range(2):
            if i == 0:
                data.nb_traces = nbt_a
                temp = False
                t = nbt_a
            else:
                data.nb_traces = nbt_p
                temp = True
                t = nbt_p
            gen_file_att(trace1, label1, s1, plaintext1, t, nbs, temp) 