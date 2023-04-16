# 适用于答题卡学号, 客观题以及主观题的结构识别服务

## Overview
* 基于YOLOv8的答题卡版面识别

## Usage
* 更改config.yaml中的字段以自定义训练/部署设置

```shell
# 启动服务
python run.py

# 训练模型
python train.py 
```

## Data Map
> vanilla output

|  id   | content  |
|  ----  | ----  |
| 0  | 学号 |
| 1  | 主观题 |
| 2  | 填空题 |
| 3  | 客观题 |

<!--1, 4, 3, 2-->

### Data Convert
> labelme2yolo

* 若重新使用labelme标注了一批新的数据，可以通过运行
```shell
python utils/labelme2yolo.py  --json_dir ... --val_size ... --output_dir ...
```
将原始的labelme格式json转换为yolo格式的图片+标记
|  arg   | help  |
|  ----  | ----  |
| json_dir  | 使用labelme标注的json目录 |
| val_size  | 分割出的验证集比例 |
| output_dir  | 转换后的保存目录 |
