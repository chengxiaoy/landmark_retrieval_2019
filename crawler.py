import requests
import os
import time


def get_local_file_exists_size(local_path):
    try:
        lsize = os.stat(local_path).st_size
    except:
        lsize = 0
    return lsize


def get_file_obj(down_link, offset):
    webPage = None
    try:
        headers = {'Range': 'bytes=%d-' % offset}
        webPage = requests.get(down_link, stream=True, headers=headers, timeout=120, verify=False)
        status_code = webPage.status_code
        if status_code in [200, 206]:
            webPage = webPage
        elif status_code == 416:
            print
            u"%s文件数据请求区间错误,status_code:%s" % (down_link, status_code)
        else:
            print
            u"%s链接有误,status_code:%s" % (down_link, status_code)
    except Exception as e:
        print
        u"无法链接:%s,e:%s" % (down_link, e)
    finally:
        return webPage


down_link = ''  # 下载链接
file_size = 271768736  # 文件总大小
local_path = "/home/adger/image/test.mp4"
while True:
    lsize = get_local_file_exists_size(local_path)
    if lsize == file_size:
        break
    webPage = get_file_obj(down_link, lsize)
    try:
        file_obj = open(local_path, 'ab+')
    except Exception as e:
        print
        u"打开文件:%s失败" % local_path
        break
    try:
        for chunk in webPage.iter_content(chunk_size=64 * 1024):
            if chunk:
                file_obj.write(chunk)
            else:
                break
    except Exception as e:
        time.sleep(5)
    file_obj.close()
    webPage.close()
