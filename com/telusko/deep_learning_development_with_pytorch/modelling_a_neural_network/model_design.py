from torchvision import models
from torchvision.models import VGG16_Weights

vgg16 = models.vgg16(weights=VGG16_Weights.DEFAULT)
print(vgg16.avgpool)
print(vgg16.classifier)