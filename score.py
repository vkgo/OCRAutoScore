import segmentation.Layout4Card.api as OuterSegmentation
import segmentation.blankSegmentation.blank_segmentation as BlankSegmentation
import scoreblocks.singleCharacterRecognition as SingleCharacterRecognition
import scoreblocks.fillblankmodel as FillBlankModel
import scoreblocks.candemo as CanDemo
import scoreblocks.essayscoremodel as EssayScoreModel
import PIL.Image
import cv2
import os

class scoresystem:
    def __init__(self):
        # 模型
        self.outer_segmentation = OuterSegmentation.OuterSegmentation()
        self.blank_segmentation = BlankSegmentation.Model()
        self.single_character_recognition = SingleCharacterRecognition.Model('../scoreblocks/CharacterRecognition/SpinalVGG_dict.pth', 'SpinalVGG')
        self.fill_blank_model = FillBlankModel.model()
        self.candemo = CanDemo.model()
        self.essay_score_model = EssayScoreModel.model()
        # 答案
        # answer是一个数组，每项是一个字典，字典格式如下：
        # {'section': 'xzt', # section的意思是题目类型，xzt是选择题，tkt是填空题，zwt是作文题
        # 'value': [...]} # value里面的值是各小题的正确答案
        self.answer = None

    def set_answer(self, answer):
        self.answer = answer


    def tkt_score(self, section_img, section_answer):
        # 2.填空分割
        blank_segmentation_result = self.blank_segmentation.process_img(section_img) # blank_segmentation_result是一个数组，每项都是图片ndarray
        score_result = {'section':'tkt'}
        right_array = []
        # 3.OCR单词识别
        for i in range(len(blank_segmentation_result)):
            recognition_result = self.fill_blank_model.recognize_text(blank_segmentation_result[i])
            if recognition_result is not None:
                if recognition_result[1] == section_answer[i]:
                    right_array.append(1)
                else:
                    judge_index = self.fill_blank_model.judge_with_clip(section_answer[i], recognition_result[1], blank_segmentation_result[i])
                    if judge_index == 0:
                        right_array.append(1)
                    else:
                        right_array.append(0)
            else:
                right_array.append(0)
        score_result['value'] = right_array
        return score_result

    def tkt_math_score(self, section_img, section_answer):
        # 2.填空分割
        blank_segmentation_result = self.blank_segmentation.process_img(
            section_img)  # blank_segmentation_result是一个数组，每项都是图片ndarray
        score_result = {'section': 'tkt_math'}
        right_array = []
        # 3.数学公式识别
        for i in range(len(blank_segmentation_result)):
            recognition_result = self.candemo.output_img(blank_segmentation_result[i])
            if recognition_result is not None:
                if recognition_result[1] == section_answer[i]:
                    right_array.append(1)
                else:
                    judge_index = self.fill_blank_model.judge_with_clip(section_answer[i], recognition_result[1], blank_segmentation_result[i])
                    if judge_index == 0:
                        right_array.append(1)
                    else:
                        right_array.append(0)
            else:
                right_array.append(0)
        score_result['value'] = right_array
        return score_result

    def zwt_score(self, section_img):
        score_result = {'section':'zwt'}
        right_array = []
        # 用ppocr获得全部英文
        essay = ''
        str_set = self.fill_blank_model.ocr.ocr(section_img)[0]
        if str_set is not None:
            for str_item in str_set:
                essay += str_item[1][0]
            # 用模型判断
            result = self.essay_score_model.getscore([essay])
            if result != None:
                result = result / 12 * 100
                right_array.append(result)
            else:
                right_array.append(0)
        else:
            right_array.append(0)
        score_result['value'] = right_array
        return score_result

    def get_score(self, img: PIL.Image.Image):
        total_result = []
        # 这个是返回的批改结果，格式为数组，每个数组元素都是一个字典，字典格式为：
        # {'section':科目, 'value':[一个01数组，1表示对应index的小题对，0表示对应index的小题错]}

        # 获取填空题答案
        answer_set_index = 0
        # 1.外框分割
        outer_segmentation_results = self.outer_segmentation.get_segmentation(img)
        CLS_ID_NAME_MAP = {
            0: 'student_id',
            1: 'subjective_problem',
            2: 'fillin_problem',
            3: 'objective_problem'
        }
        # 从results中提取出标签为3: 'objective_problem'的box，并从原图中裁剪出来，然后展示到屏幕上
        for outer_segmentation_result in outer_segmentation_results:
            for box in outer_segmentation_result.boxes:
                cls_id = box.cls.cpu().numpy()[0]
                x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                cls_name = CLS_ID_NAME_MAP[cls_id]
                if cls_name == 'student_id':
                    continue
                if cls_name == 'fillin_problem': # 填空题模型
                    for answer in self.answer[answer_set_index:]:
                        if answer['section'] == 'tkt': # 题目类型相符
                            answer_set_index = self.answer.index(answer)
                            section_answer = answer['value']
                            section_img = outer_segmentation_result.orig_img
                            section_img = section_img[int(y1):int(y2), int(x1):int(x2)]
                            score_result = self.tkt_score(section_img, section_answer)
                            total_result.append(score_result)
                        elif answer['section'] == 'tkt_math':
                            answer_set_index = self.answer.index(answer)
                            section_answer = answer['value']
                            section_img = outer_segmentation_result.orig_img
                            section_img = section_img[int(y1):int(y2), int(x1):int(x2)]
                            score_result = self.tkt_math_score(section_img, section_answer)
                            total_result.append(score_result)
                elif cls_name == 'subjective_problem':
                    for answer in self.answer[answer_set_index:]:
                        if answer['section'] == 'zwt':  # 题目类型相符
                            answer_set_index = self.answer.index(answer)
                            section_img = outer_segmentation_result.orig_img
                            section_img = section_img[int(y1):int(y2), int(x1):int(x2)]
                            score_result = self.zwt_score(section_img)
                            total_result.append(score_result)
                elif cls_name == 'objective_problem':
                    for answer in self.answer[answer_set_index:]:
                        if answer['section'] == 'xzt':  # 题目类型相符
                            answer_set_index = self.answer.index(answer)
                            section_answer = answer['value']
                            section_img = outer_segmentation_result.orig_img
                            section_img = section_img[int(y1):int(y2), int(x1):int(x2)]
                            # 涂改选择题模型
                            pass
            return total_result






if __name__ == '__main__':
    test_dir = './example_img'
    lst = os.listdir(test_dir)
    s = scoresystem()
    s.set_answer([{'section': 'tkt', 'value': ['60', '0.66', '600', 'ln4+3/2']},{'section': 'zwt'}])
    for i in lst:
        if i.endswith('.png') or i.endswith('.jpg'):
            path = os.path.join(test_dir, i)
            # img = PIL.Image.open(path)
            img = PIL.Image.open(path)
            total_result = s.get_score(img)
            print(total_result)
            break
