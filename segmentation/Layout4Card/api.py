from ultralytics import YOLO
import os
import cv2
import random


CLS_ID_NAME_MAP = {
    0: 'student_id',
    1: 'subjective_problem',
    2: 'fillin_problem',
    3: 'objective_problem'
}
class OuterSegmentation:
    def __init__(self):
        self.model = YOLO(model='./runs/detect/train3/weights/best.pt')

    def get_segmentation(self, img):
        results =  self.model.predict(source=img, imgsz=640, save=False)
        return results

if __name__ == '__main__':
    debug = True
    folder = './testdata'
    file_names = os.listdir(folder)
    random.shuffle(file_names)
    imgs = []
    for file_name in file_names:
        img_path = os.path.join(folder, file_name)
        img = cv2.imread(img_path)
        imgs += [img]
    outer_segmentation = OuterSegmentation()
    results = outer_segmentation.get_segmentation(imgs)

    # 从results中提取出标签为3: 'objective_problem'的box，并从原图中裁剪出来，然后展示到屏幕上
    for result in results:
        for box in result.boxes:
            cls_id = box.cls.cpu().numpy()[0]
            x1,y1,x2,y2 = box.xyxy.cpu().numpy()[0]
            cls_name = CLS_ID_NAME_MAP[cls_id]
            if cls_name == 'fillin_problem':
                img = result.orig_img
                img = img[int(y1):int(y2), int(x1):int(x2)]
                cv2.imshow('img', img)
                cv2.waitKey(0)