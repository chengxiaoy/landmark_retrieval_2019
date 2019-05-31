from sklearn.externals import joblib
import numpy as np
import faiss
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.externals import joblib
from multiprocessing import Pool

query_file_path = "test_gem.pkl"
index_file_path = "index_gem.pkl"

query_images, query_features = joblib.load(query_file_path)
index_images, index_features = joblib.load(index_file_path)

query_ids = [x.split("/")[-1].split(".")[0] for x in query_images]
index_ids = [x.split("/")[-1].split(".")[0] for x in index_images]

query_ids = np.array(query_ids)
index_ids = np.array(index_ids)

query_features = query_features.astype(np.float32).T
index_features = index_features.astype(np.float32).T

query_features = np.ascontiguousarray(query_features)
index_features = np.ascontiguousarray(index_features)


def get_invert_index(feature):
    res = faiss.StandardGpuResources()  # use a single GPU
    ## Using a flat index
    index_flat = faiss.IndexFlatL2(len(feature[0]))  # build a flat (CPU) index
    # make it a flat GPU index
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    gpu_index_flat.add(feature)
    return gpu_index_flat


recall_num = 1024
print(" building index")
since = time.time()
invert_index = get_invert_index(index_features)
print("build index used {} s".format(str(time.time() - since)))

print("searching...")
since = time.time()
D, I = invert_index.search(query_features, recall_num)
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
    cg_ranks = cg_diffusion(qsim, Wn, alpha)
    return cg_ranks


def rerank(I):
    dif = np.empty(I.shape)

    for i, indexs in enumerate(I):
        X = index_features[indexs].T
        Q = query_features[[i]].T
        Wn = buildGraph(X)
        rank = get_diffusion_rank(Wn, X, Q)
        indexs = indexs[rank[:, 0]]
        dif[i, :] = indexs

    I = dif
    I = I.astype(np.int32)
    return I


res = []
pool_num = 5
p = Pool(pool_num)
size = I.shape(0) // pool_num + 1
query_ids_list = []
for i in range(pool_num):
    res.append(p.apply_async(rerank, (I[i * size:(i + 1) * size, :])))
    query_ids_list.append(query_ids[i * size:(i + 1) * size])

p.close()
p.join()


f_list = []
for f in res:
    f_list.append(f.get())

with open("submit_pool_dif.csv", "w") as f:
    f.write("id,images\n")
    for future, ids in zip(f_list, query_ids_list):
        for i, indexs in enumerate(future):
            f.write(ids[i] + "," + " ".join(index_ids[indexs[0:100]]) + "\n")

with open("baseline_pool_dif.csv", "w") as f:
    for future, ids in zip(f_list, query_ids_list):
        for i, indexs in enumerate(future):
            f.write(ids[i] + "," + " ".join(index_ids[indexs]) + "\n")


