import os
import cv2
import argparse

import numpy as np
import torch
import json
import pickle as pkl
from tqdm import tqdm
import time

from utils import load_config, load_checkpoint, compute_edit_distance
from models.infer_model import Inference
from dataset import Words
from counting_utils import gen_counting_label

parser = argparse.ArgumentParser(description='model testing')
parser.add_argument('--dataset', default='CROHME', type=str, help='数据集名称')
parser.add_argument('--image_path', default='./datasets/CROHME/14_test_images.pkl', type=str, help='测试image路径')
parser.add_argument('--label_path', default='./datasets/CROHME/14_test_labels.txt', type=str, help='测试label路径')
parser.add_argument('--word_path', default='./datasets/CROHME/words_dict.txt', type=str, help='测试dict路径')

parser.add_argument('--draw_map', default=False)
args = parser.parse_args()

if not args.dataset:
    print('请提供数据集名称')
    exit(-1)

if args.dataset == 'CROHME':
    config_file = 'config.yaml'

"""加载config文件"""
params = load_config(config_file)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params['device'] = device
words = Words(args.word_path)
params['word_num'] = len(words)

if 'use_label_mask' not in params:
    params['use_label_mask'] = False
print(params['decoder']['net'])
model = Inference(params, draw_map=args.draw_map)
model = model.to(device)

load_checkpoint(model, None, params['checkpoint'])
model.eval()

with open(args.image_path, 'rb') as f:
    images = pkl.load(f)

with open(args.label_path) as f:
    lines = f.readlines()

line_right = 0
e1, e2, e3 = 0, 0, 0
bad_case = {}
model_time = 0
mae_sum, mse_sum = 0, 0

with torch.no_grad():
    for line in tqdm(lines):
        name, *labels = line.split()
        name = name.split('.')[0] if name.endswith('jpg') else name
        input_labels = labels
        labels = ' '.join(labels)
        img = images[name]
        print(np.shape(img))

        img = torch.Tensor(255-img) / 255
        img = img.unsqueeze(0).unsqueeze(0)
        img = img.to(device)
        a = time.time()
        input_labels = words.encode(input_labels)
        input_labels = torch.LongTensor(input_labels)
        input_labels = input_labels.unsqueeze(0).to(device)

        probs, _, mae, mse = model(img, input_labels, os.path.join(params['decoder']['net'], name))
        mae_sum += mae
        mse_sum += mse
        model_time += (time.time() - a)

        prediction = words.decode(probs)
        if prediction == labels:
            line_right += 1
        else:
            bad_case[name] = {
                'label': labels,
                'predi': prediction
            }
            print(name, prediction, labels)

        distance = compute_edit_distance(prediction, labels)
        if distance <= 1:
            e1 += 1
        if distance <= 2:
            e2 += 1
        if distance <= 3:
            e3 += 1

print(f'model time: {model_time}')
print(f'ExpRate: {line_right / len(lines)}')
print(f'mae: {mae_sum / len(lines)}')
print(f'mse: {mse_sum / len(lines)}')
print(f'e1: {e1 / len(lines)}')
print(f'e2: {e2 / len(lines)}')
print(f'e3: {e3 / len(lines)}')

with open(f'{params["decoder"]["net"]}_bad_case.json','w') as f:
    json.dump(bad_case,f,ensure_ascii=False)
