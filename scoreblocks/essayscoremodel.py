import configparser

import torch
import torch.nn as nn
from transformers import AutoModel
import configargparse
from transformers import AutoTokenizer
from scoreblocks.MSPLM.encoder import encode_documents
from scoreblocks.MSPLM.plms import mainplm
from torch.cuda.amp import autocast


def _initialize_arguments(p):
    p.add_section('bert_model_path')
    p.add_section('efl_encode')
    p.add_section('r_dropout')
    p.add_section('batch_size')
    p.add_section('plm_batch_size')
    p.add_section('cuda')
    p.add_section('device')
    p.add_section('test_file')
    p.add_section('data_sample_rate')
    p.add_section('prompt')
    p.add_section('chunk_sizes')
    p.add_section('train_epoch')
    p.add_section('lr_0')
    p.add_section('lr_1')
    p.add_section('w1')
    p.add_section('w2')
    p.add_section('w3')
    p.add_section('PLM')

    # p.add('--bert_model_path', help='bert_model_path')
    # p.add('--efl_encode', action='store_true', help='is continue training')
    # p.add('--r_dropout', help='r_dropout', type=float)
    # p.add('--batch_size', help='batch_size', type=int)
    # p.add('--plm_batch_size', help='plm_batch_size', type=int)
    # p.add('--cuda', action='store_true', help='use gpu or not')
    # p.add('--device')
    # p.add('--test_file', help='test data file')
    # p.add('--data_sample_rate', help='data_sample_rate', type=float)
    # p.add('--prompt', help='prompt')
    # p.add('--chunk_sizes', help='chunk_sizes', type=str)
    # p.add('--train_epoch', help='train_epoch', type=int)
    # p.add('--lr_0', help='lr_0', type=float)
    # p.add('--lr_1', help='lr_1', type=float)
    # p.add('--w1', help='w1', type=float)
    # p.add('--w2', help='w2', type=float)
    # p.add('--w3', help='w3', type=float)
    # p.add('--PLM', help='PLM', type=str)

    # args = p.parse_args()

    # 将ConfigParser对象转换为字典
    config_dict = {}
    for section in p.sections():
        section_dict = {}
        for option in p.options(section):
            section_dict[option] = p.get(section, option)
        config_dict[section] = section_dict
    config_dict = config_dict['p1']
    print(config_dict)

    if torch.cuda.is_available() and config_dict['cuda']:
        config_dict['device'] = 'cuda'
    else:
        config_dict['device'] = 'cpu'
    return config_dict

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(7)
class model:
    def __init__(self):
        # initialize arguments
        #p = configargparse.ArgParser(default_config_files=["../scoreblocks/MSPLM/ini/p1.ini"])
        # 创建ConfigParser对象
        p = configparser.ConfigParser()
        # 读取ini文件
        p.read("../scoreblocks/MSPLM/ini/p1.ini", encoding='utf-8')
        self.args = _initialize_arguments(p)
        print(f"device:{self.args['device']} torch_version:{torch.__version__}")

        # if args is not None:
        #     self.args = vars(args)

        self.tokenizer = AutoTokenizer.from_pretrained(self.args['plm'])
        self.prompt = int(self.args["prompt"][1])
        self.chunk_sizes = []
        self.bert_batch_sizes = []
        # 载入模型
        self.bert_regression_by_word_document = mainplm(self.args)
        self.bert_regression_by_word_document.load_state_dict(torch.load('../scoreblocks/MSPLM/prediction/p1/val0test1/best_total.bin'))
        self.bert_regression_by_word_document.to(self.args['device'])
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
    text = """@PERCENT1 of people agree that computers make life less complicated. I also agree with this. Using computers teaches hand-eye coordination, gives people the ability to learn about faraway places and people, and lets people talk online with other people. I think that these are all very important. Why wouldn't you want to have strong hand-eye coordination? I think this a very important skill. Computers help teach hand-eye coordination and they keep it strong. While you're looking at the screen your hand is moving the mouse where you want it to go. Good hand-eye coordination is used for a lot of things; mostly everything. If you play some sports like baseball, hand-eye is one of the most important elements. Why not make that stronger off of the feild? Also, hand-eye can be used to @ORGANIZATION1 while taking notes. Hand-eye is involved with almost everything you do. you can't have a poor hand-eye coordination or else you won't be able to function properly. @NUM1 out of @NUM2 doctors agree that hand-eye very important for healthy living. I love to travel, but I want to know about the place I'm going to before I get on the phone to go there." said @PERSON1, a science teacher at @ORGANIZATION1. He feels the way, I'm sure, a lot of people feel. They want to know about the place they are going to and they want it to be current. The computer has plenty information about a lot of different places in the world. Some books don't offer as much information or they need to be updated. Computers are also very good for learning about other cultures and traditions. No one wants to be ignorant right? People want to know what's going on in the world quick and easy. The computer does this. I remember when I was about @NUM2, our phone broke in our house. We couldn't go out and get one right away either. The only way we were able to communicate with our family and friends was by computer. The computer made it easier to e-mail everyone and tell them why we weren't answering our house phone. This happens more often than you think. People need to communicate through computer a lot. At work, if you need to talk to an employee or co-worker and you can't leave your desk, you can just e-mail the information to them. @NUM4 out of @NUM2 employees say that it is much faster and easier to e-mail information as opposed to talking them on the phone or in person. A lot of people agree that computer make life a lot easier. Computers teach hand-eye coordination and they let you communicate with other people. The most critical reason is that computers let people learn about faraway places and people. You can make a difference in the way people feel about computers. Write to your local newspaper. It's now or never!"""
    valdata = ['我是一个好孩子', '我是一个坏孩子', text]
    print(model.getscore(valdata))