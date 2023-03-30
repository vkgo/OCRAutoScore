import os
import cv2
import argparse
import torch
import cv2
import json
import pickle as pkl
from tqdm import tqdm
import time
import matplotlib.pyplot
from PIL import Image
import numpy as np

from utils import load_config, load_checkpoint, compute_edit_distance
from models.infer_model import Inference
from dataset import Words
parser = argparse.ArgumentParser(description='model testing')
parser.add_argument('--dataset', default='CROHME', type=str, help='数据集名称')
parser.add_argument('--image_path', default='./samples', type=str, help='测试image路径')
parser.add_argument('--label_path', default='./demo_labels.txt', type=str, help='测试label路径')
parser.add_argument('--word_path', default='./datasets/CROHME/words_dict.txt', type=str, help='测试dict路径')
parser.add_argument('--draw_map', default=False)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args=parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES']='0'

config_file='config.yaml'
params=load_config(config_file)
params['device'] = device
words = Words(args.word_path)
params['word_num'] = len(words)
model=Inference(params,draw_map=False)
model=model.to(device)
load_checkpoint(model,None,'./checkpoints/demo.pth')
model.eval()

line_right = 0
e1, e2, e3 = 0, 0, 0
bad_case = {}
model_time = 0
mae_sum, mse_sum = 0, 0
with open(args.label_path) as f:
    lines=f.readlines()

with torch.no_grad():
    for line in tqdm(lines):
        # print(os.path.join(args.image_path,l))
        name,*labels=line.split()
        name=name.split('.')[0] if name.endswith('jpg') else name
        input_labels=labels
        labels=' '.join(labels)
        img_index=name+'.jpg'

        # image=Image.open(os.path.join(args.image_path,img_index))
        image=cv2.imread(os.path.join(args.image_path,img_index))

        image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.asarray(image)

        print(np.shape(image))
        # 对数组进行维度处理（根据具体的神经网络测试需求进行维度调整）
        # processed_img_array = np.expand_dims(image, axis=0)
        image=torch.Tensor(255-image)/255
        image=image.unsqueeze(0).unsqueeze(0)
        image=image.to(device)
        input_labels = words.encode(input_labels)
        input_labels = torch.LongTensor(input_labels)
        input_labels = input_labels.unsqueeze(0).to(device)
        probs, _, mae, mse = model(image, input_labels, os.path.join(params['decoder']['net'], name))
        prediction = words.decode(probs)
        print(prediction)
        mae_sum += mae
        mse_sum += mse