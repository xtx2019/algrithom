import requests
import os
import urllib

if __name__ == '__main__':
    data = []
    with open("图片链接.txt", "r") as fh:
        s = fh.readlines()
        num = 1
        for line in s:
            data.append(line.strip("\n"))
            link = str(line)
            if num >= 853:
                url = link  # 图片路径。
                print(url)
                dir = './RSICD_images/'  # 当前工作目录。
                urllib.request.urlretrieve(url, dir + str(num) + '.jpg')  # 下载图片。
            num = num + 1
