import torch
from transformers import AutoTokenizer
from plms import mainplm, chunkplm
from evaluate import evaluation
from encoder import encode_documents
from data import asap_essay_lengths, fix_score
from lossfunctions import multi_loss
import pandas as pd
import matplotlib.pyplot as plt
import math
from torch.cuda.amp import autocast, GradScaler

class AESmodel():
    def __init__(self, traindata, valdata, testdata, foldname, args=None):
        if args is not None:
            self.args = vars(args)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args['PLM'])
        self.prompt = int(args.prompt[1])
        chunk_sizes_str = self.args['chunk_sizes']
        self.chunk_sizes = []
        self.bert_batch_sizes = []
        if "0" != chunk_sizes_str:
            for chunk_size_str in chunk_sizes_str.split("_"):
                chunk_size = int(chunk_size_str)
                self.chunk_sizes.append(chunk_size)
                # -------------------
                bert_batch_size = int(asap_essay_lengths[self.prompt] / chunk_size) + 1
                # -------------------
                self.bert_batch_sizes.append(bert_batch_size) # the number of chunk used in each chunksize's bert
        plm_batch_size_str = ",".join([str(item) for item in self.bert_batch_sizes])

        print("prompt:%d, asap_essay_length:%d" % (self.prompt, asap_essay_lengths[self.prompt]))
        print("chunk_sizes_str:%s, plm_batch_size_str:%s" % (chunk_sizes_str, plm_batch_size_str))


        # self.mainplm = mainplm(self.args)
        # self.chunkplm = chunkplm(self.args)
        self.bert_regression_by_word_document = mainplm(self.args)
        self.bert_regression_by_chunk = chunkplm(self.args)

        self.multi_loss = multi_loss(self.args)
        self.lr = [self.args['lr_0'], self.args['lr_1']]
        self.optim = torch.optim.Adam([
                {'params': self.bert_regression_by_word_document.parameters(), 'lr': self.lr[0]},
                {'params': self.bert_regression_by_chunk.parameters(), 'lr': self.lr[1]}
            ])

        self.traindata = traindata
        self.valdata = valdata
        self.testdata = testdata
        self.foldname = foldname

        # these are used to plot the Training Curve Chart
        self.plt_x = []
        self.plt_train_qwk = []
        self.plt_val_qwk = []
        self.plt_test_qwk = []
        self.best_val_qwk = 0.

    def adjust_learning_rate(self, epoch, start_lr, min_lr=1e-6):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        optimizer = self.optim
        lr_0 = max(start_lr[0] * (0.9 ** epoch), min_lr)
        lr_1 = max(start_lr[1] * (0.9 ** epoch), min_lr)
        optimizer.param_groups[0]['lr'] = lr_0
        optimizer.param_groups[1]['lr'] = lr_1
        print(f'{lr_0}\t{lr_1}')

    def adjust_loss_weight(self, e):
        cosvalue = max((math.cos((e / (self.args['train_epoch'] * 0.8)) * math.pi) + 1) / 2, 1e-3)
        self.multi_loss.weight = [self.args['w1'],
                                  self.args['w2'],
                                  self.args['w3'] * cosvalue]

    def validate(self, valdata, e=-1, mode='val'):
        self.bert_regression_by_word_document.eval()
        self.bert_regression_by_chunk.eval()
        scaler = GradScaler()
        with torch.no_grad():
            target_scores = None
            if isinstance(valdata, tuple) and len(valdata) == 2:
                doctok_token_indexes, doctok_token_indexes_slicenum = encode_documents(
                    valdata[0], self.tokenizer, max_input_length=512)
                # [document_number:144, 510times:3, 3, bert_len:512] [每document有多少510:144]
                # traindata[0] is the essays
                chunk_token_indexes_list, chunk_token_indexes_length_list = [], []
                for i in range(len(self.chunk_sizes)): # 以固定的chunk划分
                    document_representations_chunk, document_sequence_lengths_chunk = encode_documents(
                        valdata[0],
                        self.tokenizer,
                        max_input_length=self.chunk_sizes[i])
                    chunk_token_indexes_list.append(document_representations_chunk)
                    chunk_token_indexes_length_list.append(document_sequence_lengths_chunk)
                target_scores = torch.FloatTensor(valdata[1])

            predictions = torch.empty((doctok_token_indexes.shape[0]))
            acculation_loss = 0.
            for i in range(0, doctok_token_indexes.shape[0], self.args['batch_size']):  # range(0, 144, 32)
                batch_doctok_token_indexes = doctok_token_indexes[i:i + self.args['batch_size']].to(
                    device=self.args['device'])
                batch_target_scores = target_scores[i:i + self.args['batch_size']].to(device=self.args['device'])
                with autocast():
                    batch_doctok_predictions = self.bert_regression_by_word_document(batch_doctok_token_indexes,
                                                                                 device=self.args['device'])
                batch_doctok_predictions = torch.squeeze(batch_doctok_predictions)

                batch_predictions = batch_doctok_predictions
                # for chunk_index in range(len(self.chunk_sizes)):
                #     batch_document_tensors_chunk = chunk_token_indexes_list[chunk_index][
                #                                    i:i + self.args['batch_size']].to(
                #         device=self.args['device'])
                #     batch_predictions_chunk = self.bert_regression_by_chunk(
                #         batch_document_tensors_chunk,
                #         device=self.args['device'],
                #         plm_batch_size=self.bert_batch_sizes[chunk_index]
                #     )
                #     batch_predictions_chunk = torch.squeeze(batch_predictions_chunk)
                #     batch_predictions = torch.add(batch_predictions, batch_predictions_chunk)  # 多个chunk的分加起来
                if len(batch_predictions.shape) == 0: # 证明只有一个tensor，不构成list
                    batch_predictions = torch.tensor([batch_predictions], device=self.args['device'])
                with autocast():
                    loss = self.multi_loss(batch_target_scores.unsqueeze(1), batch_predictions.unsqueeze(1))
                acculation_loss += loss.item()

                predictions[i:i + self.args['batch_size']] = batch_predictions
            assert target_scores.shape == predictions.shape
            print(f'valset avg loss is {acculation_loss / doctok_token_indexes.shape[0]}')

            prediction_scores = []
            label_scores = []
            predictions = predictions.detach().numpy()
            target_scores = target_scores.detach().numpy()

            for index, item in enumerate(predictions):
                prediction_scores.append(fix_score(item, self.prompt))
                label_scores.append(target_scores[index])

            train_eva_res = evaluation(label_scores, prediction_scores)
            df = pd.DataFrame(dict(zip(['prediction', 'prediction_fix', 'target'],
                                       [predictions.tolist(), prediction_scores, label_scores])))
            df.to_csv(f'./prediction/p{self.prompt}/{self.foldname}/{mode}/{e + 1}_pred.csv', index=False)
            print('-' * 10 + f'{mode}set' + '-' * 10)
            print("pearson:", float(train_eva_res[7]))
            print("qwk:", float(train_eva_res[8]))
            if mode == 'val':
                self.plt_val_qwk.append(float(train_eva_res[8]))
                if self.best_val_qwk < float(train_eva_res[8]):
                    self.best_val_qwk = float(train_eva_res[8])
            elif mode == 'test':
                self.plt_test_qwk.append(float(train_eva_res[8]))
                if self.best_val_qwk == self.plt_val_qwk[-1]:
                    # save model
                    torch.save(self.bert_regression_by_word_document.state_dict(), f'./prediction/p{self.prompt}/{self.foldname}/best_total.bin')
                    torch.save(self.bert_regression_by_chunk.state_dict(), f'./prediction/p{self.prompt}/{self.foldname}/best_chunk.bin')
                    with open(f'./prediction/p{self.prompt}/{self.foldname}/best_epoch.txt', 'w') as f:
                        f.write(f'epoch {e + 1} val_qwk {self.best_val_qwk} test_qwk {float(train_eva_res[8])}')



    def train(self):
        epoch = self.args['train_epoch']
        traindata = self.traindata
        # device
        self.bert_regression_by_word_document.to(device=self.args['device'])
        self.bert_regression_by_chunk.to(device=self.args['device'])
        self.multi_loss.to(device=self.args['device'])
        scaler = GradScaler()
        for e in range(epoch):
            print('*' * 20 + f'epoch: {e + 1}' + '*' * 20)
            self.adjust_learning_rate(e, self.lr)
            self.adjust_loss_weight(e)
            self.bert_regression_by_word_document.train()
            self.bert_regression_by_chunk.train()
            target_scores = None
            if isinstance(traindata, tuple) and len(traindata) == 2:
                doctok_token_indexes, doctok_token_indexes_slicenum = encode_documents(
                    traindata[0], self.tokenizer, max_input_length=512)
                # [document_number:144, 510times:3, 3, bert_len:512] [每document有多少510:144]
                # traindata[0] is the essays
                chunk_token_indexes_list, chunk_token_indexes_length_list = [], []
                for i in range(len(self.chunk_sizes)): # 以固定的chunk划分
                    document_representations_chunk, document_sequence_lengths_chunk = encode_documents(
                        traindata[0],
                        self.tokenizer,
                        max_input_length=self.chunk_sizes[i])
                    chunk_token_indexes_list.append(document_representations_chunk)
                    chunk_token_indexes_length_list.append(document_sequence_lengths_chunk)
                target_scores = torch.FloatTensor(traindata[1])


            predictions = torch.empty((doctok_token_indexes.shape[0]))
            acculation_loss = 0.
            for i in range(0, doctok_token_indexes.shape[0], self.args['batch_size']): # range(0, 144, 32)
                self.optim.zero_grad()
                batch_doctok_token_indexes = doctok_token_indexes[i:i + self.args['batch_size']].to(device=self.args['device'])
                batch_target_scores = target_scores[i:i + self.args['batch_size']].to(device=self.args['device'])
                with autocast():
                    batch_doctok_predictions = self.bert_regression_by_word_document(batch_doctok_token_indexes, device=self.args['device'])
                batch_doctok_predictions = torch.squeeze(batch_doctok_predictions)


                batch_predictions = batch_doctok_predictions
                # for chunk_index in range(len(self.chunk_sizes)):
                #     batch_document_tensors_chunk = chunk_token_indexes_list[chunk_index][i:i + self.args['batch_size']].to(
                #         device=self.args['device'])
                #     batch_predictions_chunk = self.bert_regression_by_chunk(
                #         batch_document_tensors_chunk,
                #         device=self.args['device'],
                #         plm_batch_size=self.bert_batch_sizes[chunk_index]
                #     )
                #     batch_predictions_chunk = torch.squeeze(batch_predictions_chunk)
                #     batch_predictions = torch.add(batch_predictions, batch_predictions_chunk) # 多个chunk的分加起来

                if len(batch_predictions.shape) == 0: # 证明只有一个tensor，不构成list
                    batch_predictions = torch.tensor([batch_predictions], device=self.args['device'])
                with autocast():
                    loss = self.multi_loss(batch_target_scores.unsqueeze(1), batch_predictions.unsqueeze(1))
                
                
                
                
                loss.requires_grad_(True)
                # loss.backward()
                scaler.scale(loss).backward()
                
                # self.optim.step()
                scaler.step(self.optim)
                scaler.update()
                acculation_loss += loss.item()

                predictions[i:i + self.args['batch_size']] = batch_predictions
            assert target_scores.shape == predictions.shape

            print(f'epoch{e + 1} avg loss is {acculation_loss / doctok_token_indexes.shape[0]}')
            # 到此已获得predictions
            prediction_scores = []
            label_scores = []
            predictions = predictions.detach().numpy()
            target_scores = target_scores.detach().numpy()

            for index, item in enumerate(predictions):
                prediction_scores.append(fix_score(item, self.prompt))
                label_scores.append(target_scores[index])

            train_eva_res = evaluation(label_scores, prediction_scores)
            df = pd.DataFrame(dict(zip(['prediction', 'prediction_fix', 'target'], [predictions.tolist(), prediction_scores, label_scores])))
            df.to_csv(f'./prediction/p{self.prompt}/{self.foldname}/train/{e + 1}_pred.csv', index=False)
            print('-' * 10 + 'trainset' + '-' * 10)
            print("pearson:", float(train_eva_res[7]))
            print("qwk:", float(train_eva_res[8]))
            self.plt_x.append(e + 1)
            self.plt_train_qwk.append(float(train_eva_res[8]))
            self.validate(self.valdata, e, mode='val')
            self.validate(self.testdata, e, mode='test')
            plt.plot(self.plt_x, self.plt_train_qwk, 'ro-', color='blue', alpha=0.8, linewidth=1, label='train')
            plt.plot(self.plt_x, self.plt_val_qwk, 'ro-', color='yellow', alpha=0.8, linewidth=1, label='val')
            plt.plot(self.plt_x, self.plt_test_qwk, 'ro-', color='red', alpha=0.8, linewidth=1, label='test')
            plt.title(self.foldname)
            plt.xlabel('epoch')
            plt.ylabel('qwk')
            plt.legend(loc='lower right')
            plt.savefig(f'./prediction/p{self.prompt}/{self.foldname}/qwk.jpg')
            plt.close()