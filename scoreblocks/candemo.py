import torch
import numpy as np
from CAN.utils import load_config, load_checkpoint, compute_edit_distance
from CAN.dataset import Words
import sys
# sys.path.append('./CAN/')#
from CAN.models.infer_model import Inference
import cv2
class model:
    def __init__(self):
        # self.model_name=name
        # self.model=torch.load(path)
        self.params=load_config('../CAN/config.yaml')
        # params['device'] = device
        self.words = Words('../scoreblocks/CAN/words_dict.txt')
        self.params['word_num'] = len(self.words)
        self.params['device']='cpu'
        if 'use_label_mask' not in self.params:
            self.params['use_label_mask'] = False
        # print(params['decoder']['net'])
        self.model = Inference(self.params, draw_map=False)
        load_checkpoint(self.model,None,'../CAN/checkpoints/demo.pth')
        # self.model.cpu()
    def output(self,img_path):
        img=cv2.imread(img_path)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image = np.asarray(image)

        # print(np.shape(image))
        # 对数组进行维度处理（根据具体的神经网络测试需求进行维度调整）
        # processed_img_array = np.expand_dims(image, axis=0)
        image = torch.Tensor(255 - image) / 255
        image = image.unsqueeze(0).unsqueeze(0)
        pre,_, mae, mse=self.model(image,None,None)
        pre=self.words.decode(pre)
        print(pre)
        return pre

    def output_img(self, img):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image = np.asarray(image)

        # print(np.shape(image))
        # 对数组进行维度处理（根据具体的神经网络测试需求进行维度调整）
        # processed_img_array = np.expand_dims(image, axis=0)
        image = torch.Tensor(255 - image) / 255
        image = image.unsqueeze(0).unsqueeze(0)
        pre, _, mae, mse = self.model(image, None, None)
        pre = self.words.decode(pre)
        # print(pre)
        return pre

if __name__ == '__main__':
    model=model()
    image_path='./scoreblocks/CAN/samples/test_15.jpg'
    model.output(img_path=image_path)