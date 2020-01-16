## 安装依赖
- docker环境：jimmylin1017cs/darknet:latest    
- docker服务器：58.58.111.158 
- 镜像中含代码，但代码较乱，建议通过挂载覆盖
```
nvidia-docker run -it runtime=nvidia --name=darknet-yolo -v /home/zhangzhao/project/git/darknet:/jimmy/ jimmylin1017cs/darknet:latest /bin/bash
```
### 数据准备 
1. 数据标注参考`labelimg`，生成`jpg`对应的`xml`
2. 文件目录组织规范`cd ~/data/daozha(类别目录)`
```
|-->train_imgs(训练集图片)
    |-->原始图像(*.jpg)
|-->eval_imgs(验证集图片)
    |-->原始图像(*.jpg)
|-->train_anns （训练集标注）
    |-->(*.xml)
|-->eval_anns（验证集标注）
    |-->(*.xml)
|-->annotations(darknet训练使用)
    |-->train_anns
        |-->(*.txt,*.jpg)
    |-->eval_anns
        |-->(*.txt,*.jpg)
    |-->(*.txt)
|-->cfg(模型文件与label name列表)
    |-->(*.cfg,voc.names)
```
3. 在annotations目录中生成darknet训练使用的数据
```
python scripts/voc_label.py -k daozha
```

### 训练参数准备
见 [darknet-训练自己的yolov3模型](https://www.cnblogs.com/pprp/p/9525508.html)

### 使用说明

#### GPU训练
```
./darknet detector train <data_cfg> <model_cfg> <weights> -gpus <gpu_list>

例如：
./darknet detector train data/voc.data data/daozha/cfg/yolov3-voc.cfg backup/init/darknet53.conv.74 -gpus 2
```

#### 测试-c
```
./darknet detect cfg/yolov3-tiny.cfg yolov3-tiny.weights data/dog.jpg
```
#### 测试-python
```
python python/darknet.py
```

### darknet yolov3 转 caffe yolov3
见 [caffe frcnn](https://gitee.com/tutu123456/dashboard/programs/27934/projects/tutu123456/caffe-frcnn/code/)
