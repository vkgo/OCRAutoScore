from  torch.utils.data import random_split

class fivefold:
    def __init__(self, dataset):
        """
        data's type is Dataset
        """
        self.essay_folds = []
        self.score_folds = []
        fold_length = len(dataset) // 5
        fold_last_length = len(dataset) - (len(dataset) // 5) * 4
        subsets = random_split(dataset=dataset, lengths=[fold_length, fold_length, fold_length, fold_length, fold_last_length])
        for subset in subsets:
            essays = []
            scores = []
            for id, essay, score, prediction_id in subset:
                essays.append(essay)
                scores.append(score)
            self.essay_folds.append(essays)
            self.score_folds.append(scores)

if __name__ == '__main__':
    """
    Here is for testing.
    """
    from asap.makedataset import Dataset
    import pickle
    import matplotlib.pyplot as plt
    with open(f'./asap/pkl/train/p1_dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
    folds = fivefold(dataset)
    for scores in folds.score_folds:
        plt.plot(range(2, 13), [scores.count(i) / len(scores) for i in range(2, 13)], color='blue')
    plt.show()
    plt.close()

    valessays = []
    valscores = []
    testessays = []
    testscores = []
    trainessays = []
    trainscores = []
    for val_index in range(len(folds.essay_folds)):
        for test_index in range(len(folds.essay_folds)):
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
            # 计算分布
            plt.plot(range(2, 13), [trainscores.count(i) / len(trainscores) for i in range(2, 13)], color='blue')
            plt.plot(range(2, 13), [testscores.count(i) / len(testscores) for i in range(2, 13)], color='yellow')
            plt.plot(range(2, 13), [valscores.count(i) / len(valscores) for i in range(2, 13)], color='red')
            plt.show()
            plt.close()
    pass