import torch
import torchvision
import cv2
import numpy as np


class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)
        
    def forward(self, x):
        feature     = self.backbone(x)
        probability = torch.softmax(feature, dim=1)
        return probability
        

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

image = cv2.imread("workspace/kej.jpg")
image = cv2.resize(image, (224, 224))            # resize
image = image[..., ::-1]                         # BGR -> RGB
image = image / 255.0
image = (image - imagenet_mean) / imagenet_std   # normalize
image = image.astype(np.float32)                 # float64 -> float32
image = image.transpose(2, 0, 1)                 # HWC -> CHW
image = np.ascontiguousarray(image)              # contiguous array memory
image = image[None, ...]                         # CHW -> 1CHW
image = torch.from_numpy(image)                  # numpy -> torch
model = Classifier().eval()

with torch.no_grad():
    probability   = model(image)
    
predict_class = probability.argmax(dim=1).item()
confidence    = probability[0, predict_class]

labels = open("workspace/labels.imagenet.txt").readlines()
labels = [item.strip() for item in labels]

print(f"Predict: {predict_class}, {confidence}, {labels[predict_class]}")

dummy = torch.zeros(1, 3, 224, 224)
torch.onnx.export(
    model, (dummy,), "workspace/classifier.onnx", 
    input_names=["image"], 
    output_names=["prob"], 
    dynamic_axes={"image": {0: "batch"}, "prob": {0: "batch"}},
    opset_version=11
)