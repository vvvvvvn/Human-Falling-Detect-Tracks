import os
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import requests
from lxml import etree
from requests import get, head


def download_file(url, filename):
    req = requests.get(url)
    if req.status_code != 200:
        print("Download Error!")
        return
    try:
        with open(filename, "wb") as f:
            f.write(req.content)
            print("Download Success")
    except Exception as e:
        print(e)


class downloader:
    """from https://blog.csdn.net/qq_42951560/article/details/108785802"""

    def __init__(self, url, num, name):
        self.url = url
        self.num = num
        self.name = name
        self.getsize = 0
        r = head(self.url, allow_redirects=True)
        self.size = int(r.headers["Content-Length"])

    def down(self, start, end, chunk_size=10240):
        headers = {"range": f"bytes={start}-{end}"}
        r = get(self.url, headers=headers, stream=True)
        with open(self.name, "rb+") as f:
            f.seek(start)
            for chunk in r.iter_content(chunk_size):
                f.write(chunk)
                self.getsize += chunk_size

    def main(self):
        start_time = time.time()
        f = open(self.name, "wb")
        f.truncate(self.size)
        f.close()
        tp = ThreadPoolExecutor(max_workers=self.num)
        futures = []
        start = 0
        for i in range(self.num):
            end = int((i + 1) / self.num * self.size)
            future = tp.submit(self.down, start, end)
            futures.append(future)
            start = end + 1
        while True:
            process = self.getsize / self.size * 100
            last = self.getsize
            time.sleep(1)
            curr = self.getsize
            down = (curr - last) / 1024
            if down > 1024:
                speed = f"{down/1024:6.2f}MB/s"
            else:
                speed = f"{down:6.2f}KB/s"
            print(f"process: {process:6.2f}% | speed: {speed}", end="\r")
            if process >= 100:
                print(f"process: {100.00:6}% | speed:  00.00KB/s", end=" | ")
                break
        tp.shutdown()
        end_time = time.time()
        total_time = end_time - start_time
        average_speed = self.size / total_time / 1024 / 1024
        print(f"total-time: {total_time:.0f}s | average-speed: {average_speed:.2f}MB/s")


if __name__ == "__main__":
    # some params
    thread_num = 128

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.105 Safari/537.36"
    }
    url = "http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html"
    page_text = requests.get(url=url, headers=headers).text
    tree = etree.HTML(page_text)

    # Fall sequences
    # tr[3-32], td[2-8] (not download the 8-th file)
    # if not os.path.exists("./Fall_sequences"):
    #     os.mkdir("Fall_sequences")
    # for i in range(3, 33):
    #     rootpath = "./Fall_sequences/{}".format(i - 2)
    #     if not os.path.exists(rootpath):
    #         os.mkdir(rootpath)
    #     for j in range(2, 8):
    #         link = tree.xpath(
    #             "/html/body/div[3]/table/tr[{}]/td[{}]/a/@href".format(i, j)
    #         )
    #         link = link[0]
    #         link_name = tree.xpath(
    #             "/html/body/div[3]/table/tr[{}]/td[{}]/a/text()".format(i, j)
    #         )
    #         file_path = os.path.join(rootpath, link_name[0])
    #         # print(link, file_path)
    #         if 2 <= j <= 5:
    #             down = downloader(link, thread_num, file_path)
    #             down.main()
    #             with zipfile.ZipFile(file_path) as zf:
    #                 zf.extractall(rootpath)
    #             os.remove(file_path)
    #         elif 6 <= j <= 7:
    #             down = downloader(link, thread_num, file_path)
    #             down.main()
    #         else:
    #             print("Error!")

    # Activities of Daily Living (ADL) sequences:
    # tr[3-42], td[2-8] (not download the 8-th file)
    if not os.path.exists("./ADL_sequences"):
        os.mkdir("ADL_sequences")
    for i in range(34, 45):
        if i == 22 or i == 43:
            continue
        elif i < 22:
            rootpath = "./ADL_sequences/{}".format(i - 2)
            if not os.path.exists(rootpath):
                os.mkdir(rootpath)
        elif 22 < i < 43:
            rootpath = "./ADL_sequences/{}".format(i - 3)
            if not os.path.exists(rootpath):
                os.mkdir(rootpath)
        elif i == 44:
            rootpath = "./ADL_sequences/{}".format(i - 4)
            if not os.path.exists(rootpath):
                os.mkdir(rootpath)
        for j in [2, 4, 6, 7]:
            link = tree.xpath(
                "/html/body/div[4]/table/tr[{}]/td[{}]/a/@href".format(i, j)
            )
            link = link[0]
            link_name = tree.xpath(
                "/html/body/div[4]/table/tr[{}]/td[{}]/a/text()".format(i, j)
            )
            file_path = os.path.join(rootpath, link_name[0])
            # print(link, file_path)
            if 2 <= j <= 5:
                down = downloader(link, thread_num, file_path)
                down.main()
                with zipfile.ZipFile(file_path) as zf:
                    zf.extractall(rootpath)
                os.remove(file_path)
            elif 6 <= j <= 7:
                down = downloader(link, thread_num, file_path)
                down.main()
            else:
                print("Error!")
