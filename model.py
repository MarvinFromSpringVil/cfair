import torch
from torchvision import models, transforms
from PIL import Image

# 사전 학습된 VGG16 모델 불러오기
model = models.vgg16(pretrained=True)  # 'pretrained=True'는 사전 학습된 가중치 사용
model.eval()  # 평가 모드로 전환 (학습이 아니라 추론만 할 거니까)

# 모델 확인
print(model)