import os
import pickle
from PIL import Image

# img = Image.open('example.jpg')  # 读取图片
imgs=[]
for i in os.listdir('../samples'):
    if i.endswith('.jpg'):
        imgs.append(Image.open(i))


with open('sample.pkl', 'wb') as f:
    pickle.dump(imgs, f)  # 序列化并将图片写入pkl文件