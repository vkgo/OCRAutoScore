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
        self.model = YOLO(model='../segmentation/Layout4Card/runs/detect/train3/weights/best.pt')

    def get_segmentation(self, img):
        results =  self.model.predict(source=img, imgsz=640, save=False)
        return results

if __name__ == '__main__':
    debug = True
    folder = './testdata'
    file_names = os.listdir(folder)
    save_folder = './saved'
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    batch_size = 4
    random.shuffle(file_names)
    imgs = []
    results = []
    outer_segmentation = OuterSegmentation()
    for i in range(0, len(file_names), batch_size):
        batch_file_name = file_names[i:i+batch_size]
        for file_name in batch_file_name:
            img_path = os.path.join(folder, file_name)
            img = cv2.imread(img_path)
            imgs += [img]
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
                    # 保存img到目标文件夹，文件名随机生成
                    cv2.imwrite(os.path.join(save_folder, str(random.randint(0, 1000000))+'.jpg'), img)