import numpy as np
import h5py as h


def gen_file_att(trace1, label1, secret1,p1 , nbt, nbs, prof : bool):
    f = h.File("./resultat/test_batch.h5", "a")
    
    if prof == False:
        grp1 = f.create_group("Attack_traces")
    else:
        grp1 = f.create_group("Profiling_traces")
    
    traces1 = grp1.create_dataset("traces", data = trace1, chunks = True, maxshape=(nbt,nbs))
    
    labels1 = grp1.create_dataset("labels", data = label1, chunks = True ,maxshape=(nbt,nbs))
    
    metadata_type = np.dtype([("plaintext", p1.dtype, (len(p1[0]),)),
                              ("key", secret1.dtype, (len(secret1),))
                              ])

    attack_metadata = np.array([(p1[n], secret1) for n in range(len(p1))], dtype=metadata_type)

    metadata1 = grp1.create_dataset("metadata", data = attack_metadata, dtype = metadata_type, chunks = True, maxshape=(nbt,))

    f.close()

def gen_add(trace1, label1, secret1,p1, prof: bool):
    f = h.File("./resultat/test_batch.h5", "r+")

    if prof == False:
        traces = f["/Attack_traces/traces"]
        labels = f["/Attack_traces/labels"]
        metadata = f["/Attack_traces/metadata"]
    else:
        traces = f["/Profiling_traces/traces"]
        labels = f["/Profiling_traces/labels"]
        metadata = f["/Profiling_traces/metadata"]
    
    traces.resize(traces.shape[0] + trace1.shape[0], axis= 0)
    traces [-trace1.shape[0]:, ] = trace1

    
    labels.resize(labels.shape[0] + label1.shape[0], axis= 0)
    labels [-label1.shape[0]:, ] = label1

    metadata_type = np.dtype([("plaintext", p1.dtype, (len(p1[0]),)),
                                ("key", secret1.dtype, (len(secret1),))
                                ])

    attack_metadata = np.array([(p1[n], secret1) for n in range(len(p1))], dtype=metadata_type)

    

    
    metadata.resize(metadata.shape[0] + attack_metadata.shape[0], axis = 0)
    metadata[-attack_metadata.shape[0]:, ] = attack_metadata
    
    f.close()