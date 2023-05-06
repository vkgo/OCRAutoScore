import os
import cv2
import torch
import torch.nn as nn
import math

from CAN.models.densenet import DenseNet
from CAN.models.attention import Attention
from CAN.models.decoder import PositionEmbeddingSine
from CAN.models.counting import CountingDecoder as counting_decoder
from CAN.counting_utils import gen_counting_label
from CAN.utils import draw_attention_map, draw_counting_map


class Inference(nn.Module):
    def __init__(self, params=None, draw_map=False):
        super(Inference, self).__init__()
        self.params = params
        self.draw_map = draw_map
        self.use_label_mask = params['use_label_mask']
        self.encoder = DenseNet(params=self.params)
        self.in_channel = params['counting_decoder']['in_channel']
        self.out_channel = params['counting_decoder']['out_channel']
        self.counting_decoder1 = counting_decoder(self.in_channel, self.out_channel, 3)
        self.counting_decoder2 = counting_decoder(self.in_channel, self.out_channel, 5)
        self.device = params['device']
        self.decoder = decoder_dict[params['decoder']['net']](params=self.params)

        """经过cnn后 长宽与原始尺寸比缩小的比例"""
        self.ratio = params['densenet']['ratio']
        with open(params['word_path']) as f:
            words = f.readlines()
            print(f'共 {len(words)} 类符号。')
        self.words_index_dict = {i: words[i].strip() for i in range(len(words))}
        self.cal_mae = nn.L1Loss(reduction='mean')
        self.cal_mse = nn.MSELoss(reduction='mean') 

    def forward(self, images, labels, name, is_train=False):
        cnn_features = self.encoder(images)
        batch_size, _, height, width = cnn_features.shape
        counting_preds1, counting_maps1 = self.counting_decoder1(cnn_features, None)
        counting_preds2, counting_maps2 = self.counting_decoder2(cnn_features, None)
        counting_preds = (counting_preds1 + counting_preds2) / 2
        counting_maps = (counting_maps1 + counting_maps2) / 2

        if labels == None:
            mae, mse = None, None
        else:
            mae = self.cal_mae(counting_preds, gen_counting_label(labels, self.out_channel, True)).item()
            mse = math.sqrt(self.cal_mse(counting_preds, gen_counting_label(labels, self.out_channel, True)).item())

        word_probs, word_alphas = self.decoder(cnn_features, counting_preds, is_train=is_train)

        if self.draw_map:
            if not os.path.exists(os.path.join(self.params['attention_map_vis_path'], name)):
                os.makedirs(os.path.join(self.params['attention_map_vis_path'], name), exist_ok=True)
            if not os.path.exists(os.path.join(self.params['counting_map_vis_path'], name)):
                os.makedirs(os.path.join(self.params['counting_map_vis_path'], name), exist_ok=True)
            for i in range(images.shape[0]):
                img = images[i][0].detach().cpu().numpy() * 255
                # draw attention_map
                for step in range(len(word_probs)):
                    word_atten = word_alphas[step][0].detach().cpu().numpy()
                    word_heatmap = draw_attention_map(img, word_atten)
                    cv2.imwrite(os.path.join(self.params['attention_map_vis_path'], name, f'word_{step}.jpg'), word_heatmap)
                # draw counting_map
                for idx in range(self.out_channel):
                    counting_map = counting_maps[0].permute(1,2,0)[:,:,idx].detach().cpu()
                    counting_heatmap = draw_counting_map(img, counting_map)
                    img_name = 'symbol_' + self.words_index_dict[idx] + '_map.jpg'
                    cv2.imwrite(os.path.join(self.params['counting_map_vis_path'], name, img_name), counting_heatmap)

        return word_probs, word_alphas, mae, mse


class AttDecoder(nn.Module):
    def __init__(self, params):
        super(AttDecoder, self).__init__()
        self.params = params
        self.input_size = params['decoder']['input_size']
        self.hidden_size = params['decoder']['hidden_size']
        self.out_channel = params['encoder']['out_channel']
        self.attention_dim = params['attention']['attention_dim']
        self.dropout_prob = params['dropout']
        self.device = params['device']
        self.word_num = params['word_num']
        self.ratio = params['densenet']['ratio']

        self.init_weight = nn.Linear(self.out_channel, self.hidden_size)
        self.embedding = nn.Embedding(self.word_num, self.input_size)
        self.word_input_gru = nn.GRUCell(self.input_size, self.hidden_size)
        self.encoder_feature_conv = nn.Conv2d(self.out_channel, self.attention_dim, kernel_size=1)
        self.word_attention = Attention(params)

        self.word_state_weight = nn.Linear(self.hidden_size, self.hidden_size)
        self.word_embedding_weight = nn.Linear(self.input_size, self.hidden_size)
        self.word_context_weight = nn.Linear(self.out_channel, self.hidden_size)
        self.counting_context_weight = nn.Linear(self.word_num, self.hidden_size)
        self.word_convert = nn.Linear(self.hidden_size, self.word_num)

        if params['dropout']:
            self.dropout = nn.Dropout(params['dropout_ratio'])

    def forward(self, cnn_features, counting_preds, is_train=False):
        batch_size, _, height, width = cnn_features.shape
        image_mask = torch.ones((batch_size, 1, height, width)).to(self.device)
        
        cnn_features_trans = self.encoder_feature_conv(cnn_features)
        position_embedding = PositionEmbeddingSine(256, normalize=True)
        pos = position_embedding(cnn_features_trans, image_mask[:,0,:,:])
        cnn_features_trans = cnn_features_trans + pos

        word_alpha_sum = torch.zeros((batch_size, 1, height, width)).to(device=self.device)
        hidden = self.init_hidden(cnn_features, image_mask)
        word_embedding = self.embedding(torch.ones([batch_size]).long().to(device=self.device))
        counting_context_weighted = self.counting_context_weight(counting_preds)
        word_probs = []
        word_alphas = []

        i = 0
        while i < 200:
            hidden = self.word_input_gru(word_embedding, hidden)
            word_context_vec, word_alpha, word_alpha_sum = self.word_attention(cnn_features, cnn_features_trans, hidden,
                                                                               word_alpha_sum, image_mask)

            current_state = self.word_state_weight(hidden)
            word_weighted_embedding = self.word_embedding_weight(word_embedding)
            word_context_weighted = self.word_context_weight(word_context_vec)
            
            if self.params['dropout']:
                word_out_state = self.dropout(current_state + word_weighted_embedding + word_context_weighted + counting_context_weighted) 
            else:
                word_out_state = current_state + word_weighted_embedding + word_context_weighted + counting_context_weighted

            word_prob = self.word_convert(word_out_state)
            _, word = word_prob.max(1)
            word_embedding = self.embedding(word)
            if word.item() == 0:
                return word_probs, word_alphas
            word_alphas.append(word_alpha)
            word_probs.append(word)
            i+=1
        return word_probs, word_alphas

    def init_hidden(self, features, feature_mask):
        average = (features * feature_mask).sum(-1).sum(-1) / feature_mask.sum(-1).sum(-1)
        average = self.init_weight(average)
        return torch.tanh(average)


decoder_dict = {
    'AttDecoder': AttDecoder
}