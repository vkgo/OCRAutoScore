# OCRAutoScore

此仓库为大创集成仓库，用于将各模块按照流程图一个个完成，并留下输入输出相关接口、文档。

## 1 总流程图

![系统流程图](README.assets/系统流程图.jpg)

除了小题分割、大题分割需要等待数据集，现在已经可以开始做其他模块了。

## 2 模块开发规范

示例，如`scoreblocks/fillblankmodel.py`文件一样，写一个类，实际操作中，我们实例化这个类，然后在这个类的`init`中加载各使用到的模型（后面就不用每次调用都要加载了），然后使用类中各个成员函数来实现功能。而py文件中的`if __name__ == "__main__":`可以用来测试用。类、函数的注释中英文都行。

```python
import paddleocr
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

debug = False

class model:
    def __init__(self, language:str="en"):
        """
        :parameter language: the language of the text, `ch`, `en`, `french`, `german`, `korean`, `japan`, type: str
        """
        self.ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang=language)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", device=self.device)

    def recognize_text(self, _img:Image):
        """
        Predict the text from image
        :parameter img: image, type: np.ndarray
        :return: result, type: tuple{location: list, text: str}
        """
        img = np.array(_img)
        result = self.ocr.ocr(img)
        if debug:
            print(result)
        if len(result[0]) == 0:
            return None
        else:
            location = result[0][0][0]
            text = result[0][0][1][0]
            return (location, text)

    def judge_with_clip(self, _answer:str, _predict:str, _img:Image):
        """
        Use clip to judge which one is more similar to the Image
        :parameter answer: the answer text, type: str
        :parameter predict: the predict text, type: str
        :parameter img: image, type: np.ndarray
        :return: result, the index of the more similar text, type: int
        """
        image = _img
        inputs = self.clip_processor(text=[f"A picture with the text \"{_answer}\"", f"A picture with the text \"{_predict}\"",
                                 "A picture with the other text"], images=image, return_tensors="pt", padding=True)
        inputs.to(self.device)

        outputs = self.clip_model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        if debug:
            print(probs)
        index = torch.argmax(probs, dim=1)
        return index

if __name__ == "__main__":
    """
    用于测试函数
    """
    debug = True
    import paddle
    print(paddle.device.is_compiled_with_cuda())
    model = model()
    while True:
        img_path = input("请输入图片路径: ")
        answer = input("请输入正确答案: ")
        img = Image.open(img_path)
        predict = model.recognize_text(img)[1]
        print("预测结果: ", predict)
        if (predict != answer):
            print("正确结果：", answer)
            index = model.judge_with_clip(answer, predict, img)
            print("判断结果: ", (answer, predict, "error")[index])
```

## 3 文件目录说明

```shell
OCRAutoScore
+----scoreblocks # 填空题、选择题、作文的批改模型文件夹
|    CharacterRecognition
|    |    +----SpinalVGG.pth # SpinalVGG模型
|    |    +----WaveMix.pth # WaveMix模型
|    |    +----example # 测试图片
|    fillblank_testdata # 填空题测试图片
|    MSPLM # AES模型 
|    +----essayscoremodel.py # 作文评分模型
|    +----fillblankmodel.py # 填空题批改模型
|    +----singleCharacterRecognition.py # 单字母识别模型
+----README.assets # README的图片文件夹
+----README.md # 仓库说明文件
+----.gitignore # git忽略的文件夹、文件
```

# 4 大题分割





# 5 小题分割





# 6 选择题模型





# 7 填空题模型-中英文识别





# 8 填空题模型-公式识别
# 8.1 框架
![公式识别CAN框架](README.assets/CAN.png）



# 9 作文评分模型





# 10 WebUI

