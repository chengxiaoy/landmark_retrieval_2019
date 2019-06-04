from sklearn.externals import joblib
import numpy as np
import faiss
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
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


recall_num = 5
weigths = np.diag(np.logspace(0, -1.5, recall_num))

print(" building index")
since = time.time()
invert_index = get_invert_index(index_features)
print("build index used {} s".format(str(time.time() - since)))

print("searching for query vect ...")
since = time.time()
D, I = invert_index.search(query_features, recall_num)
print("search used {} s".format(str(time.time() - since)))
augment_query_features = np.zeros(query_features.shape)

for i, index in enumerate(I):
    augment_query_features[i, :] = query_features[i] + np.sum(np.dot(weigths, index_features[index]), axis=0)

augment_query_features = normalize(augment_query_features)

print("searching for index vect ...")
since = time.time()
D, I = invert_index.search(index_features, recall_num)
print("search used {} s".format(str(time.time() - since)))
augment_index_features = np.zeros(index_features.shape)

for i, index in enumerate(I):
    augment_index_features[i, :] = index_features[i] + np.sum(np.dot(weigths, index_features[index]), axis=0)

augment_index_features = normalize(augment_index_features)

joblib.dump((query_ids, augment_query_features, index_ids, augment_index_features), "dba_2.pkl")
