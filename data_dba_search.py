from sklearn.externals import joblib
import numpy as np
import faiss
import time

query_ids, augment_query_features, index_ids, augment_index_features = joblib.load("dba_2.pkl")

augment_query_features = augment_query_features.astype(np.float32)
augment_index_features = augment_index_features.astype(np.float32)

augment_query_features = np.ascontiguousarray(augment_query_features)
augment_index_features = np.ascontiguousarray(augment_index_features)


def get_flat_index(feature):
    res = faiss.StandardGpuResources()  # use a single GPU
    ## Using a flat index
    index_flat = faiss.IndexFlatL2(len(feature[0]))  # build a flat (CPU) index
    # make it a flat GPU index
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    gpu_index_flat.add(feature)
    return gpu_index_flat


def get_invert_index(feature):
    nlist = 2000
    d = len(feature[0])
    quantizer = faiss.IndexFlatL2(d)  # the other index
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    # here we specify METRIC_L2, by default it performs inner-product search
    # index = faiss.index_factory(d, "IVF1000,PQ128")
    assert not index.is_trained
    index.train(feature)
    assert index.is_trained
    index.add(feature)
    index.nprobe = 500
    return index


recall_num = 3000
print(" building index")
since = time.time()
invert_index = get_flat_index(augment_index_features)
print("build index used {} s".format(str(time.time() - since)))

print("searching...")
since = time.time()
D, I = invert_index.search(augment_query_features, recall_num)
print("search used {} s".format(str(time.time() - since)))



from diffussion import *


def buildGraph(X):
    K = 50  # approx 50 mutual nns

    A = np.dot(X.T, X)
    W = sim_kernel(A).T
    W = topK_W(W, K)
    Wn = normalize_connection_graph(W)
    return Wn


def get_diffusion_rank(Wn, X, Q):
    QUERYKNN = 5
    alpha = 0.9
    # perform search
    print("begin dot")
    sim = np.dot(X.T, Q)
    qsim = sim_kernel(sim).T

    sortidxs = np.argsort(-qsim, axis=1)
    for i in range(len(qsim)):
        qsim[i, sortidxs[i, QUERYKNN:]] = 0

    qsim = sim_kernel(qsim)
    # fast_spectral_ranks = fsr_rankR(qsim, Wn, alpha, 2000)
    # return fast_spectral_ranks
    cg_ranks = cg_diffusion(qsim, Wn, alpha)
    return cg_ranks


dif = np.empty(I.shape)

for i, indexs in enumerate(I):
    X = augment_index_features[indexs].T
    Q = augment_query_features[[i]].T
    Wn = buildGraph(X)
    rank = get_diffusion_rank(Wn, X, Q)
    indexs = indexs[rank[:, 0]]
    dif[i, :] = indexs

I = dif
I = I.astype(np.int32)

with open("submit_dba_dif_512.csv", "w") as f:
    f.write("id,images\n")
    for i, indexs in enumerate(I):
        f.write(query_ids[i] + "," + " ".join(index_ids[indexs[0:100]]) + "\n")
