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

'''
for result in results:

    for box in result.boxes:
        cls_id = box.cls.cpu().numpy()[0]
        x,y,w,h = box.xywh.cpu().numpy()[0]


        cls_name = CLS_ID_NAME_MAP[cls_id]
'''

