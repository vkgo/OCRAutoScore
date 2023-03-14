import torch
import csv
import pickle

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.id = []
        self.essay = []
        self.score = []
        self.prediction_id = []

    def __len__(self):
        return len(self.essay)

    def __getitem__(self, i):
        id = self.id[i]
        essay = self.essay[i]
        score = self.score[i]
        prediction_id = self.prediction_id[i]

        return id, essay, score, prediction_id



if __name__ == '__main__':
    p1_dataset = Dataset()
    p2_d1_dataset = Dataset()
    p2_d2_dataset = Dataset()
    p3_dataset = Dataset()
    p4_dataset = Dataset()
    p5_dataset = Dataset()
    p6_dataset = Dataset()
    p7_dataset = Dataset()
    p8_dataset = Dataset()

    p1_val_dataset = Dataset()
    p2_d1_val_dataset = Dataset()
    p2_d2_val_dataset = Dataset()
    p3_val_dataset = Dataset()
    p4_val_dataset = Dataset()
    p5_val_dataset = Dataset()
    p6_val_dataset = Dataset()
    p7_val_dataset = Dataset()
    p8_val_dataset = Dataset()


    # train
    with open('./training_set_rel3-utf-8.tsv', encoding='utf-8') as tsvfile:
        text = tsvfile.read()
    lines = text.split('\n')
    for line in lines[1:]:
        item = line.split('\t')
        if len(item) == 1:
            continue
        essay_id = item[0]
        essay_set = item[1]
        essay = item[2]
        domain_1 = int(item[6])
        domain_2 = item[9]
        if essay_set == '1':
            p1_dataset.id.append(essay_id)
            p1_dataset.essay.append(essay)
            p1_dataset.score.append(domain_1)
            p1_dataset.prediction_id.append('-1')
        elif essay_set == '2':
            p2_d1_dataset.id.append(essay_id)
            p2_d1_dataset.essay.append(essay)
            p2_d1_dataset.score.append(domain_1)
            p2_d1_dataset.prediction_id.append('-1')
            p2_d2_dataset.id.append(essay_id)
            p2_d2_dataset.essay.append(essay)
            p2_d2_dataset.score.append(int(domain_2))
            p2_d2_dataset.prediction_id.append('-1')
        elif essay_set == '3':
            p3_dataset.id.append(essay_id)
            p3_dataset.essay.append(essay)
            p3_dataset.score.append(domain_1)
            p3_dataset.prediction_id.append('-1')
        elif essay_set == '4':
            p4_dataset.id.append(essay_id)
            p4_dataset.essay.append(essay)
            p4_dataset.score.append(domain_1)
            p4_dataset.prediction_id.append('-1')
        elif essay_set == '5':
            p5_dataset.id.append(essay_id)
            p5_dataset.essay.append(essay)
            p5_dataset.score.append(domain_1)
            p5_dataset.prediction_id.append('-1')
        elif essay_set == '6':
            p6_dataset.id.append(essay_id)
            p6_dataset.essay.append(essay)
            p6_dataset.score.append(domain_1)
            p6_dataset.prediction_id.append('-1')
        elif essay_set == '7':
            p7_dataset.id.append(essay_id)
            p7_dataset.essay.append(essay)
            p7_dataset.score.append(domain_1)
            p7_dataset.prediction_id.append('-1')
        elif essay_set == '8':
            p8_dataset.id.append(essay_id)
            p8_dataset.essay.append(essay)
            p8_dataset.score.append(domain_1)
            p8_dataset.prediction_id.append('-1')

    # validation
    score_dit = {}
    with open('./valid_sample_submission_2_column.csv') as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)
        for row in csv_reader:
            prediction_id = row[0]
            score = row[1]
            score_dit[prediction_id] = int(score) # type of score is int



    with open('./valid_set-utf-8.tsv', encoding='utf-8') as tsvfile:
        text = tsvfile.read()
    lines = text.split('\n')
    for line in lines[1:]:
        item = line.split('\t')
        if len(item) == 1:
            continue
        essay_id = item[0]
        essay_set = item[1]
        essay = item[2]
        domain1_predictionid = item[3]
        domain2_predictionid = item[4]
        if essay_set == '1':
            p1_val_dataset.id.append(essay_id)
            p1_val_dataset.essay.append(essay)
            p1_val_dataset.score.append(score_dit[domain1_predictionid])
            p1_val_dataset.prediction_id.append(domain1_predictionid)
        elif essay_set == '2':
            p2_d1_val_dataset.id.append(essay_id)
            p2_d1_val_dataset.essay.append(essay)
            p2_d1_val_dataset.score.append(score_dit[domain1_predictionid])
            p2_d1_val_dataset.prediction_id.append(domain1_predictionid)
            p2_d2_val_dataset.id.append(essay_id)
            p2_d2_val_dataset.essay.append(essay)
            p2_d2_val_dataset.score.append(score_dit[domain2_predictionid])
            p2_d2_val_dataset.prediction_id.append(domain2_predictionid)
        elif essay_set == '3':
            p3_val_dataset.id.append(essay_id)
            p3_val_dataset.essay.append(essay)
            p3_val_dataset.score.append(score_dit[domain1_predictionid])
            p3_val_dataset.prediction_id.append(domain1_predictionid)
        elif essay_set == '4':
            p4_val_dataset.id.append(essay_id)
            p4_val_dataset.essay.append(essay)
            p4_val_dataset.score.append(score_dit[domain1_predictionid])
            p4_val_dataset.prediction_id.append(domain1_predictionid)
        elif essay_set == '5':
            p5_val_dataset.id.append(essay_id)
            p5_val_dataset.essay.append(essay)
            p5_val_dataset.score.append(score_dit[domain1_predictionid])
            p5_val_dataset.prediction_id.append(domain1_predictionid)
        elif essay_set == '6':
            p6_val_dataset.id.append(essay_id)
            p6_val_dataset.essay.append(essay)
            p6_val_dataset.score.append(score_dit[domain1_predictionid])
            p6_val_dataset.prediction_id.append(domain1_predictionid)
        elif essay_set == '7':
            p7_val_dataset.id.append(essay_id)
            p7_val_dataset.essay.append(essay)
            p7_val_dataset.score.append(score_dit[domain1_predictionid])
            p7_val_dataset.prediction_id.append(domain1_predictionid)
        elif essay_set == '8':
            p8_val_dataset.id.append(essay_id)
            p8_val_dataset.essay.append(essay)
            p8_val_dataset.score.append(score_dit[domain1_predictionid])
            p8_val_dataset.prediction_id.append(domain1_predictionid)


    # save to pkl
    with open("./pkl/train/p1_dataset.pkl", 'wb') as f:
        pickle.dump(p1_dataset, f)
    with open("./pkl/train/p2_d1_dataset.pkl", 'wb') as f:
        pickle.dump(p2_d1_dataset, f)
    with open("./pkl/train/p2_d2_dataset.pkl", 'wb') as f:
        pickle.dump(p2_d2_dataset, f)
    with open("./pkl/train/p3_dataset.pkl", 'wb') as f:
        pickle.dump(p3_dataset, f)
    with open("./pkl/train/p4_dataset.pkl", 'wb') as f:
        pickle.dump(p4_dataset, f)
    with open("./pkl/train/p5_dataset.pkl", 'wb') as f:
        pickle.dump(p5_dataset, f)
    with open("./pkl/train/p6_dataset.pkl", 'wb') as f:
        pickle.dump(p6_dataset, f)
    with open("./pkl/train/p7_dataset.pkl", 'wb') as f:
        pickle.dump(p7_dataset, f)
    with open("./pkl/train/p8_dataset.pkl", 'wb') as f:
        pickle.dump(p8_dataset, f)


    with open("./pkl/val/p1_val_dataset.pkl", 'wb') as f:
        pickle.dump(p1_val_dataset, f)
    with open("./pkl/val/p2_d1_val_dataset.pkl", 'wb') as f:
        pickle.dump(p2_d1_val_dataset, f)
    with open("./pkl/val/p2_d2_val_dataset.pkl", 'wb') as f:
        pickle.dump(p2_d2_val_dataset, f)
    with open("./pkl/val/p3_val_dataset.pkl", 'wb') as f:
        pickle.dump(p3_val_dataset, f)
    with open("./pkl/val/p4_val_dataset.pkl", 'wb') as f:
        pickle.dump(p4_val_dataset, f)
    with open("./pkl/val/p5_val_dataset.pkl", 'wb') as f:
        pickle.dump(p5_val_dataset, f)
    with open("./pkl/val/p6_val_dataset.pkl", 'wb') as f:
        pickle.dump(p6_val_dataset, f)
    with open("./pkl/val/p7_val_dataset.pkl", 'wb') as f:
        pickle.dump(p7_val_dataset, f)
    with open("./pkl/val/p8_val_dataset.pkl", 'wb') as f:
        pickle.dump(p8_val_dataset, f)