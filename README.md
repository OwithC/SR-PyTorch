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

1. ./data/crop.py

2. ./model/two_person_model.py

3. two_person_data.py

4. two_person_main.py


## Training
1. Preparing data
```shell
python ./data/crop.py
```
2. Training
```shell
python two_person_main.py
```

## Result 

### 1) 이미지 전체
> ![image](https://user-images.githubusercontent.com/60590737/129310229-0ed8c0c4-d0e7-45e2-b900-b056dd54fa94.png)
>
> ![image](https://user-images.githubusercontent.com/60590737/129310382-f74464c4-e85e-4514-ab1e-4d3a60e826af.png)

### 2) 두 사람 영역
> ![image](https://user-images.githubusercontent.com/76933244/131054276-b893284b-8569-4535-a989-6783765c2365.png)

### 3) 결론 
```
여러 실험을 통해 데이터를 재구성한 것이 성능 측면에서 긍정적인 영향을 준다고 생각할 수 있었다.   
또한, 클래스 간의 데이터 갯수를 균등하게 설정해야 좋은 정확도를 얻을 수 있다는 것을 알 수 있었다.
```

### TODO
final release : 2021-08-27

<hr>

## Member

강보경 <mint48579@gmail.com></br>
박찬 <dkssudgkdl9@naver.com></br>

