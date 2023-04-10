import os
import segmentation.Layout4Card.api as OuterSegmentation
import segmentation.blankSegmentation.blank_segmentation as BlankSegmentation
import scoreblocks.singleCharacterRecognition as SingleCharacterRecognition
import scoreblocks.fillblankmodel as FillBlankModel
import scoreblocks.candemo as CanDemo
import scoreblocks.essayscoremodel as EssayScoreModel
import PIL.Image
class scoresystem:
    def __init__(self):
        # 模型
        self.outer_segmentation = OuterSegmentation.OuterSegmentation()
        self.blank_segmentation = BlankSegmentation.Model()
        self.single_character_recognition = SingleCharacterRecognition.Model('./scoreblocks/CharacterRecognition/SpinalVGG.pth', 'SpinalVGG')
        self.fill_blank_model = FillBlankModel.model()
        self.candemo = CanDemo.model()
        self.essay_score_model = EssayScoreModel.model()
        # 答案
        # answer是一个数组，每项是一个字典，字典格式如下：
        # {'section': 'xzt', # section的意思是题目类型，xzt是选择题，tkt是填空题，zwt是作文题
        # 'value': [...]} # value里面的值是各小题的正确答案
        self.answer = None

    def set_answer(self, answer):
        pass


    def get_score(self, img: PIL.Image.Image):
        # 1.外框分割
        outer_segmentation_result = self.outer_segmentation.get_segmentation(img)
        # 2.空白分割
        blank_segmentation_result = self.blank_segmentation.process(outer_segmentation_result)



if __name__ == '__main__':
    test_dir = './example_img'
    lst = os.listdir(test_dir)
    s = scoresystem()
    for i in lst:
        if i.endswith('.png') or i.endswith('.jpg'):
            path = os.path.join(test_dir, i)
            img = PIL.Image.open(path)
            s.get_score(img)
