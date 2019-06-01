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


recall_num = 100
print(" building index")
since = time.time()
invert_index = get_invert_index(index_features)
print("build index used {} s".format(str(time.time() - since)))

print("searching...")
since = time.time()
D, RAW_I = invert_index.search(query_features, recall_num)
I, s_I = np.split(RAW_I, [20], axis=1)

print("search used {} s".format(str(time.time() - since)))

new_querys = np.zeros(query_features.shape)
for i, index in enumerate(I):
    new_querys[i] = (query_features[i] + np.sum(index_features[index[:20]], axis=0)) / (recall_num + 1)
new_querys = new_querys.astype(np.float32)
new_querys = np.ascontiguousarray(new_querys)

I_list = []


def recursion(query_features, recall, nums):
    if nums > 20:
        return
    D, I = invert_index.search(query_features, recall)
    I_list.append(I)
    new_querys = np.zeros(query_features.shape)
    for i, index in enumerate(I):
        new_querys[i] = (query_features[i] + np.sum(index_features[index], axis=0)) / (recall_num + 1)
    new_querys = new_querys.astype(np.float32)
    new_querys = np.ascontiguousarray(new_querys)
    recursion(new_querys, recall, nums + 1)


recursion(new_querys, 20, 0)

for ii in I_list:
    I = np.concatenate((I, ii), axis=1)

I = np.concatenate((I, s_I), axis=1)

res = np.zeros((query_features.shape[0], 100))

for i, indexs in enumerate(I):
    res[i, :] = np.unique(indexs)[0:100]

I = res
I = I.astype(np.int32)

with open("submit_aqe.csv", "w") as f:
    f.write("id,images\n")
    for i, indexs in enumerate(I):
        f.write(query_ids[i] + "," + " ".join(index_ids[indexs[0:100]]) + "\n")

with open("baseline_aqe.csv", "w") as f:
    for i, indexs in enumerate(I):
        f.write(query_ids[i] + "," + " ".join(index_ids[indexs]) + "\n")
