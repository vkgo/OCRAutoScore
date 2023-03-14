from scipy.stats import pearsonr
import numpy as np


def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(rater_a, rater_b, min_rating=None, max_rating=None):

    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)

    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j] / num_scored_items)
            if num_ratings == 1:
                num_ratings += 0.0000001
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    if denominator <= 0.0000001:
        denominator = 0.0000001
    return 1.0 - numerator / denominator


def evaluation(true_label, pre_label, high_score=7.0, second_high_score=6.5, low_score=4.5, second_low_score=5.0):
    assert len(pre_label) == len(true_label)
    # 分差指标
    res = [0, 0, 0, 0, 0]
    for i in range(len(pre_label)):
        if pre_label[i] is None:
            pre_label[i] = 0
        index = int(abs(pre_label[i] - true_label[i]) / 0.5)
        if index <= 3:
            res[index] += 1
        else:
            res[4] += 1
    total_score = sum(res)
    result = [float(item) / total_score for item in res]
    result.append(result[0] + result[1])
    result.append(result[0] + result[1] + result[2])

    # pearson
    result.append(pearsonr(true_label, pre_label)[0])
    # kappa
    result.append(quadratic_weighted_kappa(true_label, pre_label))

    # high score evaluation result
    high_score_recall, high_score_precision, f1 = evaluation_high_score(true_label, pre_label, high_score)
    result.append(high_score_recall)
    result.append(high_score_precision)
    result.append(f1)

    # high score evaluation result
    second_high_score_recall, second_high_score_precision, f1 = evaluation_high_score(true_label, pre_label,
                                                                                      second_high_score)
    result.append(second_high_score_recall)
    result.append(second_high_score_precision)
    result.append(f1)

    # low score evaluation result
    second_low_score_recall, second_low_score_precision, f1 = evaluation_low_score(true_label, pre_label, second_low_score)
    result.append(second_low_score_recall)
    result.append(second_low_score_precision)
    result.append(f1)

    # low score evaluation result
    low_score_recall, low_score_precision, f1 = evaluation_low_score(true_label, pre_label, low_score)
    result.append(low_score_recall)
    result.append(low_score_precision)
    result.append(f1)

    result = [str(round(item, 3)) for item in result]
    # result_str = '|' + '|'.join(result)
    return result


def f1(precision, recall, weight=1):
    if precision == 0 or recall == 0:
        return 0
    return (weight * weight + 1) * precision * recall / (weight * weight * precision + recall)


def evaluation_high_score(true_score, pre_score, high_score):
    assert len(pre_score) == len(true_score)
    true_high_score_num = 0.0
    pred_high_score_num = 0.0
    both_high_score_num = 0.0
    qualified_num = 0.0
    smooth_value = 0.0000001
    for i in range(len(pre_score)):
        if true_score[i] >= high_score:
            true_high_score_num += 1.0
        if pre_score[i] >= high_score:
            pred_high_score_num += 1.0
        if pre_score[i] >= high_score and true_score[i] >= high_score:
            both_high_score_num += 1.0
        if pre_score[i] >= high_score and abs(pre_score[i] - true_score[i]) <= 0.5:
            qualified_num += 1.0
    high_score_recall = both_high_score_num / (true_high_score_num + smooth_value)
    high_score_precision = qualified_num / (pred_high_score_num + smooth_value)
    high_score_f1 = f1(high_score_precision, high_score_recall)
    # print(true_high_score_num, pred_high_score_num, both_high_score_num, qualified_num)
    return high_score_recall, high_score_precision, high_score_f1


def evaluation_low_score(true_score, pre_score, low_score):
    assert len(pre_score) == len(true_score)
    true_low_score_num = 0.0
    pred_low_score_num = 0.0
    both_low_score_num = 0.0
    qualified_num = 0.0
    smooth_value = 0.0000001
    for i in range(len(pre_score)):
        if true_score[i] <= low_score:
            true_low_score_num += 1.0
        if pre_score[i] <= low_score:
            pred_low_score_num += 1.0
        if pre_score[i] <= low_score and true_score[i] <= low_score:
            both_low_score_num += 1.0
        if pre_score[i] <= low_score and abs(pre_score[i] - true_score[i]) <= 0.5:
            qualified_num += 1.0
    low_score_recall = both_low_score_num / (true_low_score_num + smooth_value)
    low_score_precision = qualified_num / (pred_low_score_num + smooth_value)
    low_score_f1 = f1(low_score_precision, low_score_recall)

    return low_score_recall, low_score_precision, low_score_f1

