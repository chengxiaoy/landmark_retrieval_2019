from sklearn.externals import joblib
import numpy as np
import faiss
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from multiprocessing import Pool

query_file_path = "gem_eval_test_data.pkl"
index_file_path = "gem_eval_index_data.pkl"

query_images, query_features = joblib.load(query_file_path)
index_images, index_features = joblib.load(index_file_path)

query_ids = [x.split("/")[-1].split(".")[0] for x in query_images]
index_ids = [x.split("/")[-1].split(".")[0] for x in index_images]

query_ids = np.array(query_ids)
index_ids = np.array(index_ids)

query_features = query_features.astype(np.float32)
index_features = index_features.astype(np.float32)

# region
# print(" training pca")
# since = time.time()
# pca = PCA(n_components=2048, whiten=True)
# index_features = pca.fit_transform(index_features)
# query_features = pca.transform(query_features)
#
# index_features = normalize(index_features)
# query_features = normalize(query_features)
#
# print("traing pca used {} s".format(str(time.time() - since)))
# endregion

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

# from diffussion import *
#
#
# def buildGraph(X):
#     K = 50  # approx 50 mutual nns
#
#     A = np.dot(X.T, X)
#     W = sim_kernel(A).T
#     W = topK_W(W, K)
#     Wn = normalize_connection_graph(W)
#     return Wn
#
#
# def get_diffusion_rank(Wn, X, Q):
#     QUERYKNN = 5
#     alpha = 0.9
#     # perform search
#     print("begin dot")
#     sim = np.dot(X.T, Q)
#     qsim = sim_kernel(sim).T
#
#     sortidxs = np.argsort(-qsim, axis=1)
#     for i in range(len(qsim)):
#         qsim[i, sortidxs[i, QUERYKNN:]] = 0
#
#     qsim = sim_kernel(qsim)
#     cg_ranks = cg_diffusion(qsim, Wn, alpha)
#     return cg_ranks
#
#
# dif = np.empty(I.shape)
#
# for i, indexs in enumerate(I):
#     X = index_features[indexs].T
#     Q = query_features[[i]].T
#     Wn = buildGraph(X)
#     rank = get_diffusion_rank(Wn, X, Q)
#     indexs = indexs[rank[:, 0]]
#     dif[i, :] = indexs
#
# I = dif
# I = I.astype(np.int32)

with open("submit_base.csv", "w") as f:
    f.write("id,images\n")
    for i, indexs in enumerate(I):
        f.write(query_ids[i] + "," + " ".join(index_ids[indexs[0:100]]) + "\n")

with open("baseline_base.csv", "w") as f:
    for i, indexs in enumerate(I):
        f.write(query_ids[i] + "," + " ".join(index_ids[indexs]) + "\n")
