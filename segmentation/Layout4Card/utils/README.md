# Labelme2YOLO

Help converting LabelMe Annotation Tool JSON format to YOLO text file format. 
If you've already marked your segmentation dataset by LabelMe, it's easy to use this tool to help converting to YOLO format dataset.

## Parameters Explain
**--json_dir** LabelMe JSON files folder path.

**--val_size (Optional)** Validation dataset size, for example 0.2 means 20% for validation and 80% for training.

**--json_name (Optional)** Convert single LabelMe JSON file.

## How to Use

### 1. Convert JSON files, split training and validation dataset by --val_size
Put all LabelMe JSON files under **labelme_json_dir**, and run this python command.
```bash
python labelme2yolo.py --json_dir /home/username/labelme_json_dir/ --val_size 0.2
```
Script would generate YOLO format dataset labels and images under different folders, for example,
```bash
/home/username/labelme_json_dir/YOLODataset/labels/train/
/home/username/labelme_json_dir/YOLODataset/labels/val/
/home/username/labelme_json_dir/YOLODataset/images/train/
/home/username/labelme_json_dir/YOLODataset/images/val/

/home/username/labelme_json_dir/YOLODataset/dataset.yaml
```

### 2. Convert JSON files, split training and validation dataset by folder
If you already split train dataset and validation dataset for LabelMe by yourself, please put these folder under labelme_json_dir, for example,
```bash
/home/username/labelme_json_dir/train/
/home/username/labelme_json_dir/val/
```
Put all LabelMe JSON files under **labelme_json_dir**. 
Script would read train and validation dataset by folder.
Run this python command.
```bash
python labelme2yolo.py --json_dir /home/username/labelme_json_dir/
```
Script would generate YOLO format dataset labels and images under different folders, for example,
```bash
/home/username/labelme_json_dir/YOLODataset/labels/train/
/home/username/labelme_json_dir/YOLODataset/labels/val/
/home/username/labelme_json_dir/YOLODataset/images/train/
/home/username/labelme_json_dir/YOLODataset/images/val/

/home/username/labelme_json_dir/YOLODataset/dataset.yaml
```

### 3. Convert single JSON file
Put LabelMe JSON file under **labelme_json_dir**. , and run this python command.
```bash
python labelme2yolo.py --json_dir /home/username/labelme_json_dir/ --json_name 2.json
```
Script would generate YOLO format text label and image under **labelme_json_dir**, for example,
```bash
/home/username/labelme_json_dir/2.text
/home/username/labelme_json_dir/2.png
```

##
Only tested on Centos 7/Python 3.6 environment.
