from flask import Flask, request, render_template, jsonify
#import torch
#from torchvision import transforms, models 
#from PIL import Image
import os

app = Flask(__name__)

# PyTorch 모델 로드 (예: 학습된 모델 파일 'model.pt' 사용)
#model = torch.load('model.pt', map_location=torch.device('cpu'))  # 학습한 모델 파일 경로
'''
model = models.vgg16(pretrained=True)  # 'pretrained=True'는 사전 학습된 가중치 사용
model.eval()  # 평가 모드로 전환

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 모델 입력 크기에 맞게 조정
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
'''
# 클래스 이름 (예시, 학습한 모델에 맞게 수정)
class_names = ['고양이', '강아지', '새']  # 네 모델의 출력 클래스



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_name = data['image']
    #image_path = os.path.join('static', image_name)

    # 이미지 로드 및 전처리
    #image = Image.open(image_path).convert('RGB')
    #image_tensor = transform(image).unsqueeze(0)  # 배치 차원 추가

    # 모델 예측
    #with torch.no_grad():
    #    output = model(image_tensor)
    #    _, predicted = torch.max(output, 1)
    #    #result = class_names[predicted.item()]
    #    result = predicted.item()
    result = 'image_name'
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=10000)
