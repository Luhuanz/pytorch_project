
### 检测
```bash
python detect.py --source 0  # webcam
                          img.jpg  # image
                          vid.mp4  # video
                          path/  # directory
                          'path/*.jpg'  # glob
                          'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                          'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

### 训练
#### 1、YOLOv5官方模式
```bash
python train.py --data coco128.yaml --cfg yolov5n.yaml --weights '' --batch-size 128
                                          yolov5s                                64
                                          yolov5m                                40
                                          yolov5l                                24
                                          yolov5x                                16
```
#### 2、YOLOv5+SimOTA匹配模式
```bash
python train.py --data coco128.yaml --mode ota --cfg yolov5n_ota.yaml --weights '' --batch-size 128
                                                     yolov5s_ota                                64
                                                     yolov5m_ota                                40
                                                     yolov5l_ota                                24
                                                     yolov5x_ota                                16
```
#### 3、YOLOv5+TAL_Alignment匹配模式
```bash
python train.py --data coco128.yaml --mode tal_align --cfg yolov5n_tal_align.yaml --weights '' --batch-size 128
                                                           yolov5s_tal_align                                64
                                                           yolov5m_tal_align                                40
                                                           yolov5l_tal_align                                24
                                                           yolov5x_tal_align                                16
```
#### 4、YOLOv5+ObjectBox匹配模式
```bash
python train.py --data coco128.yaml --mode objectBox --cfg yolov5n_objectBox.yaml --weights '' --batch-size 128
                                                           yolov5s_objectBox                                64
                                                           yolov5m_objectBox                                40
                                                           yolov5l_objectBox                                24
                                                           yolov5x_objectBox                                16
```



```bash
# Single-GPU
python classify/train.py --model yolov5s-cls.pt --data cifar100 --epochs 5 --img 224 --batch 128

# Multi-GPU DDP
python -m torch.distributed.run --nproc_per_node 4 --master_port 1 classify/train.py --model yolov5s-cls.pt --data imagenet --epochs 5 --img 224 --device 0,1,2,3
```

### Val
Validate accuracy on a pretrained model. To validate YOLOv5s-cls accuracy on ImageNet.
```bash
bash data/scripts/get_imagenet.sh --val  # download ImageNet val split (6.3G, 50000 images)
python classify/val.py --weights yolov5s-cls.pt --data ../datasets/imagenet --img 224
```

### Predict
Run a classification prediction on an image.
```bash
python classify/predict.py --weights yolov5s-cls.pt --data data/images/bus.jpg
```
```python
model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5s-cls.pt')  # load from PyTorch Hub
```

### Export
Export a group of trained YOLOv5-cls, ResNet and EfficientNet models to ONNX and TensorRT.
```bash
python export.py --weights yolov5s-cls.pt resnet50.pt efficientnet_b0.pt --include onnx engine --img 224
```
</details>


## <div align="center">Contribute</div>
