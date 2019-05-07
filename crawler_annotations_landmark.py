import os
import requests
import multiprocessing
from PIL import Image
from io import BytesIO
import hashlib


def parse_md5(s):
    m = hashlib.md5()

    # Tips
    # 此处必须encode
    # 若写法为m.update(str)  报错为： Unicode-objects must be encoded before hashing
    # 因为python3里默认的str是unicode
    # 或者 b = bytes(str, encoding='utf-8')，作用相同，都是encode为bytes
    b = s.encode(encoding='utf-8')
    m.update(b)
    str_md5 = m.hexdigest()
    return str_md5


def download_image(url, out_dir):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36',
    }

    proxies = {
        "http": "socks5://127.0.0.1:1080",
        'https': 'socks5://127.0.0.1:1080'
    }

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    key = parse_md5(url)

    filename = os.path.join(out_dir, '%s.jpg' % key)
    if os.path.exists(filename):
        print('Image %s already exists. Skipping download.' % filename)
        return filename
    try:
        r = requests.get(url, stream=True, headers=headers, timeout=3, proxies=proxies, verify=False)
        # r = requests.get(url, stream=True, headers=headers, timeout=3)
    except:
        print('Warning: Could not download image %s from %s' % (key, url))
        return

    try:
        pil_image = Image.open(BytesIO(r.content))
    except:
        print('Warning: Failed to parse image %s' % key)
        return

    try:
        pil_image_rgb = pil_image.convert('RGB')
    except:
        print('Warning: Failed to convert image %s to RGB' % key)
        return

    try:
        pil_image_rgb.save(filename, format='JPEG', quality=90)
    except:
        print('Warning: Failed to save image %s' % filename)
        return
    return filename


def get_label_dict_from_txt(txt_file):
    f = open(txt_file, 'r')
    clean_data = f.readlines()
    f.close()

    pic_dict = {}
    for line in clean_data:
        url = line.split(" ")[0]
        label = line.split(" ")[1]
        if pic_dict.__contains__(label):
            pic_dict[label].append(url)
        else:
            pic_dict[label] = [url]
    return pic_dict


landmark_train_dict = get_label_dict_from_txt("annotations_landmarks/annotation_clean_train.txt")

landmark_val_dict = get_label_dict_from_txt("annotations_landmarks/annotation_clean_val.txt")

train_file = "annotation_clean_train.txt"
val_file = "annotation_clean_val.txt"

train_data_dict = {}


def download_train_and_record(keys):
    res_dict = {}
    for key in keys:
        urls = landmark_train_dict[key]
        paths = []
        for url in urls:
            try:
                path = download_image(url, "finetuning_landmark_train_data/")
                paths.append(path)
            except Exception as e:
                print("download url {} failed {}".format(url, e))
            res_dict[key] = paths
    return res_dict


def dict_2_file(dicts, file_path):
    lines = []
    with open(file_path, "w") as f:
        for dict in dicts:
            for key in dict:
                paths = dict[key]
                for path in paths:
                    lines.append(path + " " + key)
    f.writelines(lines)


pool_num = 100
p = multiprocessing.Pool(processes=pool_num)
keys = list(landmark_train_dict.keys())
sub_size = len(keys) // pool_num
results = []
for i in range(pool_num):
    results.append(p.apply_async(download_train_and_record, args=(keys[i * sub_size:(i + 1) * sub_size],)))
print('Waiting for all subprocesses done...')
p.close()
p.join()
print('All subprocesses done.')


dict_2_file(results, train_file)
