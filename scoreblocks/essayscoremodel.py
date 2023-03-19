import torch
import torch.nn as nn
from transformers import AutoModel
import configargparse
from transformers import AutoTokenizer
from MSPLM.encoder import encode_documents
from MSPLM.plms import mainplm
from torch.cuda.amp import autocast


def _initialize_arguments(p: configargparse.ArgParser):
    p.add('--bert_model_path', help='bert_model_path')
    p.add('--efl_encode', action='store_true', help='is continue training')
    p.add('--r_dropout', help='r_dropout', type=float)
    p.add('--batch_size', help='batch_size', type=int)
    p.add('--plm_batch_size', help='plm_batch_size', type=int)
    p.add('--cuda', action='store_true', help='use gpu or not')
    p.add('--device')
    p.add('--test_file', help='test data file')
    p.add('--data_sample_rate', help='data_sample_rate', type=float)
    p.add('--prompt', help='prompt')
    p.add('--chunk_sizes', help='chunk_sizes', type=str)
    p.add('--train_epoch', help='train_epoch', type=int)
    p.add('--lr_0', help='lr_0', type=float)
    p.add('--lr_1', help='lr_1', type=float)
    p.add('--w1', help='w1', type=float)
    p.add('--w2', help='w2', type=float)
    p.add('--w3', help='w3', type=float)
    p.add('--PLM', help='PLM', type=str)

    args = p.parse_args()

    if torch.cuda.is_available() and args.cuda:
        args.device = 'cuda'
    else:
        args.device = 'cpu'
    return args

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(7)
class model:
    def __int__(self):
        # initialize arguments
        p = configargparse.ArgParser(default_config_files=["MSPLM/ini/p1.ini"])
        args = _initialize_arguments(p)
        print(f'device:{args.device} torch_version:{torch.__version__}')

        if args is not None:
            self.args = vars(args)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args['PLM'])
        self.prompt = int(args.prompt[1])
        self.chunk_sizes = []
        self.bert_batch_sizes = []
        # 载入模型
        self.bert_regression_by_word_document = torch.load('./MSPLM/prediction/p1/val0test1/best_total.bin')
        self.bert_regression_by_word_document.eval()

        # these are used to plot the Training Curve Chart
        self.plt_x = []
        self.plt_train_qwk = []
        self.plt_val_qwk = []
        self.plt_test_qwk = []
        self.best_val_qwk = 0.

    def getscore(self, valdata):
        """"
        param: valdata: list of str [str, str, str]. 可以批处理作文

        return: list of float [float, float, float]. 作文的得分，取值范围[0, 12]
        """
        with torch.no_grad():
            target_scores = None
            doctok_token_indexes, doctok_token_indexes_slicenum = encode_documents(
                valdata, self.tokenizer, max_input_length=512)
            # [document_number:144, 510times:3, 3, bert_len:512] [每document有多少510:144]
            # traindata[0] is the essays

            predictions = torch.empty((doctok_token_indexes.shape[0]))
            acculation_loss = 0.
            for i in range(0, doctok_token_indexes.shape[0], self.args['batch_size']):  # range(0, 144, 32)
                batch_doctok_token_indexes = doctok_token_indexes[i:i + self.args['batch_size']].to(
                    device=self.args['device'])
                with autocast():
                    batch_doctok_predictions = self.bert_regression_by_word_document(batch_doctok_token_indexes,
                                                                                     device=self.args['device'])
                batch_doctok_predictions = torch.squeeze(batch_doctok_predictions)

                batch_predictions = batch_doctok_predictions
                if len(batch_predictions.shape) == 0:  # 证明只有一个tensor，不构成list
                    batch_predictions = torch.tensor([batch_predictions], device=self.args['device'])

                predictions[i:i + self.args['batch_size']] = batch_predictions


            predictions = predictions.detach().numpy()

            # 批量检测predictions，查看每一个作文的得分是否在[0, 12]之间
            for i in range(len(predictions)):
                if predictions[i] < 0:
                    predictions[i] = 0
                elif predictions[i] > 12:
                    predictions[i] = 12

            return predictions


if __name__ == '__main__':
    model = model()
    valdata = ['我是一个好孩子', '我是一个坏孩子']
    print(model.getscore(valdata))