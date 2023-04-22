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

# 4 作答区域分割-大题分割

在大题分割部分，我们使用了YOLOv8模型，通过使用老师提供的数据集进行训练，最终呈现了十分完美的效果。

## 4.1 YOLOv8

<img src="README.assets/62b5ed8625d6c4a157f1ee7a1c10c4e9.png" alt="YOLOv8来啦 | 详细解读YOLOv8的改进模块！YOLOv5官方出品YOLOv8，必卷！ - 智源社区" style="zoom: 67%;" />

YOLOv8是一个包括了图像分类、Anchor-Free物体检测和实例分割的高效算法，检测部分设计参考了目前大量优异的最新的YOLO改进算法，实现了新的SOTA。YOLOv8抛弃了前几代模型的Anchor-Base，是一种基于图像全局信息进行预测的目标检测系统。

与前代相比，YOLOv8有以下不同：

1. 提供了全新的SOTA模型，包括具有P5 640和P6 1280分辨率的目标检测网络以及基于YOLACT的实例分割模型。与YOLOv5类似，还提供了不同大小的N/S/M/L/X比例模型，根据缩放系数来满足不同场景需求。

2. 骨干网络和Neck部分可能参考了YOLOv7 ELAN的设计思路。在YOLOv5中，C3结构被替换为具有更丰富梯度流动性的C2f结构。对于不同比例模型进行了不同通道数量调整，并经过精心微调而非盲目应用一组参数到所有模型上，大大提高了模型性能。然而，在这个C2f模块中存在一些操作（如Split）并不像之前那样适合特定硬件部署。

3. 与YOLOv5相比，在Head部分进行了重大改变，采用主流解耦头结构将分类和检测头分离，并从Anchor-Based转向Anchor-Free。

4. 在损失计算策略方面，采用TaskAlignedAssigner正样本分配策略以及引入Distribution Focal Loss。

5. 在训练过程中进行数据增强时，引入自YOLOX关闭Mosaic增强操作后10个epoch可以有效提高准确性。

YOLOv8作为一种实时目标检测算法，可能被应用于多种场景，包括但不限于：

- 无人驾驶汽车：实时检测道路上的行人、车辆、交通信号等目标，为自动驾驶系统提供关键信息。
- 视频监控：实时检测和跟踪安全系统中的异常行为，如闯入、偷窃等。
- 工业自动化：用于产品质量检测、机器人导航等领域。
- 无人机航拍：实时目标检测和跟踪，为无人机提供导航和避障能力。

此处，我们将其应用于大题分割，也就是，在我们提供的整张试卷中，找到对应的大题（如：客观题、填空题、主观题等）。

## 4.2 单独执行大题分割

大题分割源码在`segmentation/Layout4Card`，也可以通过URLhttps://github.com/vkgo/OCRAutoScore/blob/3a97c0bd2b32abdeaba7c7c0bfa5106bdaee4479/segmentation/Layout4Card进入我们仓库中大题分割的目录查看、复制、运行，需要更多的支持，可以查看文档https://github.com/vkgo/OCRAutoScore/blob/aeefed4426e3088507e709cfd3cb99c891f44af2/segmentation/Layout4Card/README.md。

`infer.py`是一个推理代码的示范，在这之中：

1. CLS_ID_NAME_MAP是一个字典，里面有我们支持的识别类别和它对应的index。
2. model是载入模型，此处可以使用我们训练好的模型不需要改动。
3. folder是将要测试的图片的目录，可以换为测试图片。
4. 运行后，由于`model.predict(source=imgs, save=True, imgsz=640)`中的`save=True`，文件将被村粗在`segmentation/Layout4Card/run`目录之下。

```python
CLS_ID_NAME_MAP = {
    0: 'student_id',
    1: 'subjective_problem',
    2: 'fillin_problem',
    3: 'objective_problem'
}

model = YOLO(model='./runs/detect/train3/weights/best.pt')
folder = './testdata'
file_names = os.listdir(folder)

random.shuffle(file_names)

imgs = []
for file_name in file_names[:10]:
    img_path = os.path.join(folder, file_name)
    img = cv2.imread(img_path)
    imgs += [img]
results =  model.predict(source=imgs, save=True, imgsz=640)
```

## 4.3 样例

```shell
cd .\segmentation\Layout4Card\
python .\infer.py
```



![image-20230422110046357](README.assets/image-20230422110046357.png)

保存的图片如下：：

<img src="README.assets/image-20230422110135175.png" alt="image-20230422110135175" style="zoom: 33%;" />



# 5 作答区域分割-小题分割





# 6 选择题模型





# 7 填空题模型-中英文识别





# 8 填空题模型-公式识别
## 8.1 框架

![CAN模型框架](README.assets/CAN.png)
在本项目中，结合了Li B(2022)提出的CAN（计数感知网络），我们实现了对log、e^x等较为复杂的公式的识别。CAN整合了两部分任务：手写公式识别和符号计数。具体来说，使用了一个弱监督的符号计数模块，它可以在没有符号位置的情况下预测每个符号类的数目。
# 8.2 实现
训练部分
    在数学公式识别中，我们参考了CAN(Li B et al)使用的方法，使用注意力机制，结合encoder，decoder的方法，使用DenseNet作为encoder。在训练过程中，我们先将images输入DenseNet得到image的特征，之后，我们将该特征分别输入到预先设置的三个decoder中，前两个decoder生成counting_loss,最后一个decoder生成word_loss，通过三个loss分别训练不同的decoder。
数据集
	我们使用CROHME数据集进行公式的训练。CROHME数据集为pkl文件，我们通过python的PIL库读取该数据集。
预测
	在预测公式的过程中，预先定义一个字典，使用encoder-decoder将预测的概率与字典中的字符匹配，实现手写公式的识别，如下图所示，我们将预先定义的字符映射到字典中。
![按照字典预测](README.assets/inference.jpg)
# 8.3 运行
该项目需要pytorch1.10.2+python3.6
training：

cd scoreblocks/CAN

source activate pytorch

python train --dataset=CROHME

test:

python inference.py

# 9 作文评分模型





# 10 WebUI

