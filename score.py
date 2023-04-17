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
        self.single_character_recognition = SingleCharacterRecognition.Model('./scoreblocks/CharacterRecognition/SpinalVGG_dict.pth', 'SpinalVGG')
        self.fill_blank_model = FillBlankModel.model()
        self.candemo = CanDemo.model()
        self.essay_score_model = EssayScoreModel.model()
        # 答案
        # answer是一个数组，每项是一个字典，字典格式如下：
        # {'section': 'xzt', # section的意思是题目类型，xzt是选择题，tkt是填空题，zwt是作文题
        # 'value': [...]} # value里面的值是各小题的正确答案
        # self.answer = [{'section': 'tkt', 'value': ['60', '0.66', '600', 'ln4+3/2']}]

    def set_answer(self, answer):
       self.answer = answer


    def get_score(self, img: PIL.Image.Image):
        # 获取填空题答案
        fill_blank_answer = None
        right_number = 0
        for answer in self.answer:
            if answer['section'] == 'tkt':
                fill_blank_answer = answer['value']
                break
        # 1.外框分割
        outer_segmentation_result = self.outer_segmentation.get_segmentation(img)
        CLS_ID_NAME_MAP = {
            0: 'student_id',
            1: 'subjective_problem',
            2: 'fillin_problem',
            3: 'objective_problem'
        }
        # 从results中提取出标签为3: 'objective_problem'的box，并从原图中裁剪出来，然后展示到屏幕上
        for result in outer_segmentation_result:
            for box in result.boxes:
                cls_id = box.cls.cpu().numpy()[0]
                x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                cls_name = CLS_ID_NAME_MAP[cls_id]
                if cls_name == 'fillin_problem':
                    fill_blank_img = result.orig_img
                    fill_blank_img = fill_blank_img[int(y1):int(y2), int(x1):int(x2)]
                    # 保存img到目标文件夹，文件名随机生成
                    break

        # 2.空白分割
        blank_segmentation_result = self.blank_segmentation.process_img(fill_blank_img) # blank_segmentation_result是一个数组，每项都是图片ndarray

        # 3.OCR单词识别
        for i in range(len(blank_segmentation_result)):
            recognition_result = self.fill_blank_model.recognize_text(blank_segmentation_result[i])
            if recognition_result is not None:
                if recognition_result[1] == fill_blank_answer[i]:
                    right_number += 1
                else:
                    judge_index = self.fill_blank_model.judge_with_clip(fill_blank_answer[i], recognition_result[1], blank_segmentation_result[i])
                    if judge_index == 0:
                        right_number += 1
            pass



# if __name__ == '__main__':
#     test_dir = './example_img'
#     lst = os.listdir(test_dir)
#     s = scoresystem()
#     for i in lst:
#         if i.endswith('.png') or i.endswith('.jpg'):
#             path = os.path.join(test_dir, i)
#             img = PIL.Image.open(path)
#             s.get_score(img)
