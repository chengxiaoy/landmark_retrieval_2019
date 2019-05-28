import requests
import multiprocessing
import os

base_uri = "https://s3.amazonaws.com/google-landmark/index/"


def get_local_file_exists_size(local_path):
    try:
        lsize = os.stat(local_path).st_size
    except:
        lsize = 0
    return lsize


def get_remote_file_size(filename):
    url = base_uri + filename
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36',
    }

    proxies = {
        "http": "socks5://127.0.0.1:1080",
        'https': 'socks5://127.0.0.1:1080'
    }
    r = requests.get(url, stream=True, headers=headers, timeout=3, proxies=proxies)
    size = r.headers['content-length']
    r.close()
    return int(size)


def download_file(filename):
    url = base_uri + filename

    if os.path.exists(filename):
        rsize = get_remote_file_size(filename)
        lsize = get_local_file_exists_size(filename)
        if lsize < rsize:
            print("downloading {}".format(filename))
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36',
                'Range': 'bytes=%d-' % lsize,
            }
            proxies = {
                "http": "socks5://127.0.0.1:1080",
                'https': 'socks5://127.0.0.1:1080'
            }
            r = requests.get(url, stream=True, headers=headers, timeout=3, proxies=proxies)

            with open(filename, 'ab+') as file:
                # 每128个流遍历一次
                for data in r.iter_content(128):
                    # 把流写入到文件，这个文件最后写入完成就是，selenium.png
                    file.write(data)  # data相当于一块一块数据写入到我们的图片文件中
            r.close()
            print("downloaded {}".format(filename))
    else:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36',
        }

        proxies = {
            "http": "socks5://127.0.0.1:1080",
            'https': 'socks5://127.0.0.1:1080'
        }
        r = requests.get(url, stream=True, headers=headers, timeout=3, proxies=proxies)
        print("downloading {}".format(filename))

        with open(filename, 'wb') as file:
            # 每128个流遍历一次
            for data in r.iter_content(128):
                # 把流写入到文件，这个文件最后写入完成就是，selenium.png
                file.write(data)  # data相当于一块一块数据写入到我们的图片文件中
        r.close()
        print("downloaded {}".format(filename))


filenames = []
for i in range(100):
    index = "%03d" % i
    filename = "images_" + index + ".tar"
    filenames.append(filename)

pool_num = 5
p = multiprocessing.Pool(processes=pool_num)
p.map(download_file, filenames)
