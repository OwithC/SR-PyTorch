# SR-PyTorch

팀 OwithC - "Obession with Coding" 코딩에 대한 강박(강보경,박찬) **Social Relation** 를 소개합니다.</br>

![image](https://user-images.githubusercontent.com/60590737/129307229-1abcd312-609c-44d2-8e76-92580cd05f86.png)

1) **이미지 전체를 통해** 이미지 내의 모든 사람들의 관계를 예측하는 프로젝트와 <br/>
2) **사진 내의 두 사람 영역을 인식하고** 학습된 데이터를 기반으로 두 사람의 관계를 예측하는 프로젝트입니다.<br/> 

## The technologies used in this project
- Pytorch 1.9.0
- Python 3.8
- numpy
- cv2
- Yolov5

## Dataset 
> [PISC Dataset Download 바로가기](https://zenodo.org/record/1059155#.YRX_VHX7Q1g)

## How to install

> pytorch / CUDA Toolkit / Anaconda3 / cuDNN 8.1.0 설치

</br> pycharm 에서 프로젝트 실행

## Train the model 
```
1. ResNet
2. DenseNet
3. AlexNet
4. MobileNet
```

## Documentation

1. crop.py
  ```
  - 데이터셋 내의 이미지를 두 사람의 영역으로 crop
  - 코드를 실행하기 전에 Train, Test 로 데이터셋 나누는 작업 필요
  - annotation_image_info.json 파일 필요
  - 데이터셋, 주석 파일 경로 수정
  ```
2. two_person_model.py
```
ResNet, VGG 모델
```
3. two_person_data.py
```
relationship.json 주석 파일 필요
```
4. two_person_main.py
```
- Trian, Test로 나눈 데이터의 id 리스트를 저장하고 있는 파일 필요
- 데이터셋, 데이터 id 리스트 파일 경로 수정 필요
- checkpoint 디렉토리 생성 후 경로 수정 필요
```

## Training
```shell
python two_person_main.py
```

## Result 

### 1) 이미지 전체
> ![image](https://user-images.githubusercontent.com/60590737/129310229-0ed8c0c4-d0e7-45e2-b900-b056dd54fa94.png)
>
> ![image](https://user-images.githubusercontent.com/60590737/129310382-f74464c4-e85e-4514-ab1e-4d3a60e826af.png)

### 2) 두 사람 영역
> 1. ResNet  
> ![image](https://user-images.githubusercontent.com/76933244/129312278-4b94e438-8731-4947-b064-21cbb1f89e45.png)
> 2. VGG  
> ![image](https://user-images.githubusercontent.com/76933244/129312231-65caf57a-7506-4bd5-8172-73b90bad7004.png)

### 3) 결론 
```
사람영역만을 가지고 관계를 유추하는 것은 이미지 전체 영역에서 특징을 추출하는 것보다 성능이 좋지 못하다.
이미지 전체 및 주변 영향을 받는 것을 알 수 있다. 또한 데이터를 재구성한 것이 성능 측면에서 긍정적인 영향을 미쳤다고 생각한다. 
```

### TODO
final release : 2021-08-13

<hr>

## Member

강보경 <mint48579@gmail.com></br>
박찬 <dkssudgkdl9@naver.com></br>

