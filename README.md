## 安装依赖
- docker环境：jimmylin1017cs/darknet:latest    
- 镜像中含代码，但代码较乱，建议通过挂载覆盖
```
nvidia-docker run -it runtime=nvidia --name=darknet-yolo -v /home/zhangzhao/project/git/darknet:/jimmy/ jimmylin1017cs/darknet:latest /bin/bash
```
### 数据准备 && 训练参数准备
见 [darknet-训练自己的yolov3模型](https://www.cnblogs.com/pprp/p/9525508.html)
### 使用说明
#### 单GPU训练
```
./darknet -i <gpu_id> detector train <data_cfg> <train_cfg> <weights>

例如：
./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg darknet53.conv.74
```
#### 多GPU训练
```
./darknet detector train <data_cfg> <model_cfg> <weights> -gpus <gpu_list>

例如：
./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg darknet53.conv.74 -gpus 0,1,2,3
```
#### 测试-c
```
./darknet detect cfg/yolov3-tiny.cfg yolov3-tiny.weights data/dog.jpg
```
#### 测试-python
```
python python/darknet.py
```
