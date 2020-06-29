from sklearn.datasets import fetch_openml
from PIL import Image
import numpy as np
import ssl
import os

data_dir = "./data/"
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

#セキュリティリスク有
ssl._create_default_https_context = ssl._create_unverified_context

mnist = fetch_openml('mnist_784', version=1, data_home="./data/")


data_dir_path = "./data/img/"
if not os.path.exists(data_dir_path):
    os.mkdir(data_dir_path)

# データの取り出し
X = mnist.data
y = mnist.target
count = [0]*10
for i in range(len(X)):
    file_path="./data/img/img_"+y[i]+"_"+str(count[int(y[i])])+".jpg"
    im_f=(X[i].reshape(28, 28))  # 画像を28×28の形に変形
    pil_img_f = Image.fromarray(im_f.astype(np.uint8))  # 画像をPILに
    pil_img_f = pil_img_f.resize((64, 64), Image.BICUBIC)  # 64×64に拡大
    pil_img_f.save(file_path)  # 保存
    count[int(y[i])] += 1