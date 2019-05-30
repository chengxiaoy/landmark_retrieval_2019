import os
import sys

tar_list = []

files = os.listdir('.')
for file in files:
    if file.endswith(".tar"):
        tar_list.append(file)

if len(files) < 100:
    print("tar files num error ")

dest_path = 'pic_data/'

if not os.path.exists(dest_path):
    os.mkdir(dest_path)

for tar in tar_list:
    os.system(" tar -xvf {} -C {} ".format(tar, dest_path))
