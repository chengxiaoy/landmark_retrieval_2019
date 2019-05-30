import numpy as np
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.neighbors import BallTree
import os
from sklearn.preprocessing import normalize as sknormalize
# import crawler_img
import multiprocessing
import time
from finetuning import siames_model


def get_image_list(train_path):
    training_names = os.listdir(train_path)

    image_paths = []
    for training_name in training_names:
        path = os.path.join(train_path, training_name)

        if os.path.isdir(path):
            image_paths.extend(get_image_list(path))
        else:
            if path.endswith(".jpg"):
                image_paths += [path]
    return image_paths


class RMACJob:
    def __init__(self, model, features_path, index_tree_path, pca=False):
        self.model = model
        self.features_path = features_path
        self.index_tree_path = index_tree_path
        self.pca = pca
        if pca:
            self.pca_path = "pca.pkl"

    def batch_extract_feature(self, image_paths):
        features = []
        images = []
        for path in image_paths:
            try:
                features.append(self.model.extract_feature(path))
                images.append(path)
                print("get {} image feature success!".format(path))
            except Exception as e:
                print("get {} image feature failed!".format(path), e)
        features = np.array(features)
        return images, features

    def build_index(self):
        images, features = joblib.load(self.features_path)

        ball_tree = BallTree(features, leaf_size=30, metric='euclidean')
        joblib.dump(ball_tree, self.index_tree_path)

    def normalize(self, x, copy=False):
        """
        A helper function that wraps the function of the same name in sklearn.
        This helper handles the case of a single column vector.
        """
        if type(x) == np.ndarray and len(x.shape) == 1:
            return np.squeeze(sknormalize(x.reshape(1, -1), copy=copy))
            # return np.squeeze(x / np.sqrt((x ** 2).sum(-1))[..., np.newaxis])
        else:
            return sknormalize(x, copy=copy)
            # return x / np.sqrt((x ** 2).sum(-1))[..., np.newaxis]

    def delta_add(self, image_paths):
        """
        向图库中新增图片
        :param image_paths:
        :return:
        """
        img_paths, features = self.batch_extract_feature(image_paths)

        if self.pca:
            pca = joblib.load(self.pca_path)
            features = pca.transform(features)
            features = self.normalize(features)

        if os.path.exists(self.features_path):
            before_images, before_features = joblib.load(self.features_path)
            features = np.concatenate((before_features, features), axis=0)
            before_images.extend(img_paths)
            joblib.dump((before_images, features), self.features_path)
        else:
            joblib.dump((img_paths, features), self.features_path)

        # self.build_index()

    def get_pca_whitening(self, image_paths, n_components, whitening=False):
        _, features = self.batch_extract_feature(image_paths)

        pca = PCA(n_components, whiten=whitening)
        pca.fit(features)
        joblib.dump(pca, self.pca_path)

    def remove_relative_files(self, file_paths):
        for file_path in file_paths:
            if not file_path.startswith('/') and os.path.exists(file_path):
                os.remove(file_path)
                print("remove {} success".format(file_path))

    def get_dirs(self, exclude_max=True):
        crawler_dir = './crawler/'

        sub_dirs = os.listdir(crawler_dir)
        dir_dict = {}
        for sub_dir in sub_dirs:
            dir_index = int(sub_dir[4:])
            dir_dict[dir_index] = sub_dir
        if exclude_max:
            max_index = max(list(dir_dict.keys()))
            dir_dict.__delitem__(max_index)
        return list(dir_dict.values())

    def op_in_file(self, str, file_name='op.txt'):
        with open(file_name, 'a+') as f:
            localtime = time.asctime(time.localtime(time.time()))
            f.write(localtime + "：" + str + '\n')

    # def build_index_once(self):
    #
    #     image_paths = []
    #     image_paths.extend(get_image_list("../crawler/data_20W"))
    #
    #     self.remove_relative_files(["rmac_features.pkl", "rmac_index_tree.pkl", "rmac_pca.pkl"])
    #     self.get_pca_whitening(image_paths[0:20000], 512, True)
    #
    #     for i in range(10):
    #         # download part
    #         page_index = i
    #         page_size = 500000
    #         data = crawler_img.get_page_records(page_index, page_size)
    #         last_page = False
    #         if len(data) < page_size:
    #             last_page = True
    #         pool_num = 200
    #         p = multiprocessing.Pool(processes=pool_num)
    #         sub_size = page_size // pool_num
    #         for i in range(pool_num):
    #             p.apply_async(crawler_img.download_list, args=(data[i * sub_size:(i + 1) * sub_size], i, False,))
    #         p.close()
    #         p.join()
    #
    #         # build index part
    #         sub_dirs = self.get_dirs(not last_page)
    #         sub_dir_paths = list(map(lambda x: './crawler/' + x, sub_dirs))
    #
    #         image_paths = []
    #         for sub_dir_path in sub_dir_paths:
    #             image_paths.extend(get_image_list(sub_dir_path))
    #         self.op_in_file("build index for {} files".format(str(len(image_paths))))
    #
    #         job.delta_add(image_paths)
    #         self.op_in_file("build index for {}".format(sub_dir_paths))
    #
    #         # remove the dirs
    #         self.remove_relative_files(sub_dir_paths)
    #         self.op_in_file("remove dirs {}".format(sub_dir_paths))


if __name__ == '__main__':
    # model = RmacResNetModel()
    # job = RMACJob(model, "rmac_features.pkl", "rmac_index_tree.pkl", True)
    # job.build_index_once()

    image_paths = []
    image_paths.extend(get_image_list("index_data/pic_data"))
    print("training {} files".format(str(len(image_paths))))

    model = siames_model('SiameseNetwork(resnet50_gem_eval).pth', finetuning=True)

    job = RMACJob(model, "gem_eval_index_data.pkl", "gem_eval_test_tree.pkl", pca=False)
    job.delta_add(image_paths)


