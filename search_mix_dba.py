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

gem_test_dict = {}
for id, feature in zip(query_ids, query_features):
    gem_test_dict[id] = feature

gem_index_dict = {}
for id, feature in zip(index_ids, index_features):
    gem_test_dict[id] = feature

vgg_query_file_path = "test_vgg.pkl"
vgg_index_file_path = "index_vgg.pkl"

vgg_query_images, vgg_query_features = joblib.load(vgg_query_file_path)
vgg_index_images, vgg_index_features = joblib.load(vgg_index_file_path)

vgg_query_ids = [x.split("/")[-1].split(".")[0] for x in query_images]
vgg_index_ids = [x.split("/")[-1].split(".")[0] for x in index_images]

vgg_query_ids = np.array(vgg_query_ids)
vgg_index_ids = np.array(vgg_index_ids)

vgg_query_features = vgg_query_features.astype(np.float32).T
vgg_index_features = vgg_index_features.astype(np.float32).T

vgg_test_dict = {}
for id, feature in zip(vgg_query_ids, vgg_query_features):
    vgg_test_dict[id] = feature

vgg_index_dict = {}
for id, feature in zip(vgg_index_ids, vgg_index_features):
    vgg_test_dict[id] = feature

for qid in gem_test_dict:
    gem_test_dict[qid] = np.concatenate((gem_test_dict[qid], vgg_test_dict[qid]), axis=1)

for xid in gem_index_dict:
    gem_index_dict[xid] = np.concatenate((gem_index_dict[xid], vgg_index_dict[xid]), axis=1)

qids = list(gem_test_dict.keys())
query_features = np.zeros((query_features.shape[0],query_features.shape[1]+vgg_query_features.shape[1]))
for i,qid in enumerate(qids):
    query_features[i,:] = gem_test_dict[qid]

xids = list(gem_index_dict.keys())
index_features = np.zeros((index_features.shape[0],index_features.shape[1]+vgg_index_features.shape[1]))
for i,xid in enumerate(xids):
    index_features[i,:] = gem_index_dict[xid]


query_features = normalize(query_features)
index_features = normalize(index_features)

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


recall_num = 10
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

joblib.dump((query_ids, augment_query_features, index_ids, augment_index_features), "mix_dba.pkl")
