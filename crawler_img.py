import requests
import uuid
import pymysql
import os
import sys
from multiprocessing import Pool
import socket
import socks


def get_random_records(count):
    db = pymysql.connect("40.73.38.13", "test", "tezign", "idp")

    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()

    # 使用 execute()  方法执行 SQL 查询
    cursor.execute(
        "SELECT * FROM idp_copyright_crawler_pic WHERE id >= ((SELECT MAX(id) FROM idp_copyright_crawler_pic)-(SELECT MIN(id) FROM idp_copyright_crawler_pic)) * RAND() + (SELECT MIN(id) FROM idp_copyright_crawler_pic)  LIMIT " + str(
            count))
    # 使用 fetchone() 方法获取单条数据
    records = cursor.fetchall()
    db.close()
    return records


def get_page_records(pageIndex, pageSize):
    offset = pageIndex * pageSize
    db = pymysql.connect("40.73.38.13", "test", "tezign", "idp")

    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()

    # 使用 execute()  方法执行 SQL 查询
    cursor.execute(
        "SELECT * FROM idp_copyright_crawler_pic LIMIT " + str(offset) + "," + str(pageSize))
    # 使用 fetchone() 方法获取单条数据
    records = cursor.fetchall()
    db.close()
    return records


def get_records(offset):
    db = pymysql.connect("40.73.38.13", "test", "tezign", "idp")

    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()

    # 使用 execute()  方法执行 SQL 查询
    cursor.execute(
        "SELECT * FROM idp_copyright_crawler_pic where id >=" + str(offset))
    # 使用 fetchone() 方法获取单条数据
    records = cursor.fetchall()
    db.close()
    return records


def get_records_range(begin, end):
    db = pymysql.connect("40.73.38.13", "test", "tezign", "idp")

    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()

    # 使用 execute()  方法执行 SQL 查询
    cursor.execute(
        "SELECT * FROM idp_copyright_crawler_pic where id >= " + str(begin) + " and id<= " + str(end))
    # 使用 fetchone() 方法获取单条数据
    records = cursor.fetchall()
    db.close()
    return records


def download_image(pic, file_dir):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36',
    }

    proxies = {
        "http": "socks5://127.0.0.1:1080",
        'https': 'socks5://127.0.0.1:1080'
    }
    # socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", 1080)
    # socket.socket = socks.socksocket

    url = pic[1]
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    # items = url.split(".")
    # ext = items[len(items) - 1]
    filename = file_dir + os.path.sep + str(pic[0])
    if os.path.exists(filename):
        return
    response = requests.get(url, stream=True, headers=headers, timeout=3, proxies=proxies, verify=False)
    with open(filename, 'wb') as file:
        # 每128个流遍历一次
        for data in response.iter_content(128):
            # 把流写入到文件，这个文件最后写入完成就是，selenium.png
            file.write(data)  # data相当于一块一块数据写入到我们的图片文件中
    print(response.status_code)


def download_list(pics, idx, special_dir=False):
    for pic in pics:
        try:
            pic_id = pic[0]
            dir_path = "data"
            if not special_dir:
                dir_path = "data" + str(pic_id // 20000)
            download_image(pic, dir_path)
        except Exception as e:
            print("except:{}, subprocess {}".format(e, idx))




if __name__ == '__main__':
    # data = []
    # ranges = [(1215120, 1216119), (1677535, 1678534), (2809789, 2810788), (1, 1122), (1284589, 1285588),
    #           (1161120, 1162119), (326568, 327600), (148182, 149936), (1294968, 1295967), (1942794, 1960477)]
    # for begin,end in ranges:
    #     data.extend(get_records_range(begin,end))
    # print("get records {}".format(len(data)))
    # download_list(data, 0)
    # pass
    # url_size = 400000
    # page_index = 0
    # data = get_random_records(20000)
    # # print('Parent process %s.' % os.getpid())
    pool_num = 10
    p = Pool(pool_num)
    sub_size = len(data) // pool_num
    for i in range(pool_num):
        p.apply_async(download_list, args=(data[i * sub_size:(i + 1) * sub_size], i, True,
                                           ))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')

    #
    # proxies = {
    #     "http": "socks5://127.0.0.1:1080",
    #     'https': 'socks5://127.0.0.1:1080'
    # }
    # headers = {
    #     'User-Agent': 'Mozilla/4.0(compatible;MSIE 5.5;Windows NT)', }

    url = 'https://lh3.googleusercontent.com/-wtf3Okv9NZU/WNqxEgcY1-I/AAAAAAACWoM/W_fRiuJJhbkgzVMZSVYl2TZG3IB41VTeQCOcB/s1600/'
    download_image([132, url], "valid_data")

    # response = requests.get(url1, stream=True, headers=headers, timeout=3, proxies=proxies)
