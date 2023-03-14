import pickle
from asap.makedataset import Dataset
import torch
import configargparse
from model import AESmodel
from fivefold import fivefold
import os

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


if __name__ == "__main__":
    # initialize arguments
    p = configargparse.ArgParser(default_config_files=["ini/p1.ini"])
    args = _initialize_arguments(p)
    print(f'device:{args.device} torch_version:{torch.__version__}')
    # load dataset
    with open(f'./asap/pkl/train/{args.prompt}_dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
    folds = fivefold(dataset)

    for val_index in range(len(folds.essay_folds)):
        for test_index in range(len(folds.essay_folds)):
            valessays = []
            valscores = []
            testessays = []
            testscores = []
            trainessays = []
            trainscores = []
            if val_index == test_index:
                continue
            foldname = f'val{val_index}test{test_index}'
            for i, (essays, scores) in enumerate(zip(folds.essay_folds, folds.score_folds)):
                if i == val_index:
                    valessays = folds.essay_folds[i]
                    valscores = folds.score_folds[i]
                elif i == test_index:
                    testessays = folds.essay_folds[i]
                    testscores = folds.score_folds[i]
                else:
                    trainessays = trainessays + folds.essay_folds[i]
                    trainscores = trainscores + folds.score_folds[i]

            model = AESmodel(traindata=(trainessays, trainscores), valdata=(valessays, valscores),
                             testdata=(testessays, testscores), foldname=foldname, args=args)
            filepath = f'./prediction/{args.prompt}'
            if not os.path.isdir(filepath):
                # make dir
                os.mkdir(filepath)
            if not os.path.isdir(filepath + f'/{foldname}'):
                os.mkdir(filepath + f'/{foldname}')
                os.mkdir(filepath + f'/{foldname}/train')
                os.mkdir(filepath + f'/{foldname}/val')
                os.mkdir(filepath + f'/{foldname}/test')
            model.train()
    pass