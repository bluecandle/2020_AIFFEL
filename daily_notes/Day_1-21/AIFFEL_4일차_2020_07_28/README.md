# AIFFEL_4일차 2020.07.28

Tags: AIFFEL_DAILY_OLD

### 하루 과제

1. F2 마저 진행
2. E1 진행

### 공지사항

1. 각 과정 자체는 꼭 타임라인에 칼같이 맞춰서 할 필요는 없다.
2. 다만, Exploration 단계들을 밀리면 나중에 따라가기 힘들다.
3. 코치진 운영 방식은 이번 주는 각 반별로 다른 형태로 진행해볼 예정. (상주, 주기적 점검 등)
4. 그룹별 토론은 두 가지 형태 (상시, 정해진 시간) 로 나누어 진행해볼 예정.
5. LMS 이용 시, ~/aiffel 디렉토리로 이동한 이후 사용하는게 좋으실겁니다. (LMS 학습 내역이 거기에 저장됩니다.)

# [E-1] 인공지능과 가위바위보 하기

## 데이터셋에 대한 이해 (훈련 , 검증, 테스트)

    참고로 ‘학습’의 의미는 문제와 해답지를 같이 준 후 문제 푼 뒤 정답과 맞추어서 학습을 하라는 것이고, ‘평가’의 의미는 문제만 주고 풀게한 뒤 맞는 지 틀린 지 점수만 계산하는 것입니다. 이 과정에서는 학생이 풀이과정을 보지 않고 점수만 매기는 것과 동일하기 때문에 학습이 일어나지 않습니다.

    학생들이 스스로 학습 상태를 확인하고 학습 방법을 바꾸거나 학습을 중단하는 시점을 정할 수 없을까요? 이를 위해서 검증셋이 필요합니다. 학습할 때는 모의고사 1회~4회만 사용하고, 모의고사 5회분을 검증셋으로 두어 학습할 때는 사용하지 않습니다. 이 방식은 두 가지 효과를 얻을 수 있습니다.

첫번째로 학습 방법을 바꾼 후 훈련셋으로 학습을 해보고 검증셋으로 평가해볼 수 있습니다. 검증셋으로 가장 높은 평가를 받은 학습 방법이 최적의 학습 방법이라고 생각하면 됩니다. 이러한 학습 방법을 결정하는 파라미터를 `하이퍼파라미터(hyperparameter)`라고 하고 최적의 학습 방법을 찾아가는 것을 하이퍼파라미터 튜닝이라고 합니다.

`검증셋이 있다면 스스로 평가하면서 적절한 학습방법을 찾아볼 수 있습니다.`

두번째로 얼마정도 반복 학습이 좋을 지를 정하기 위해서 검증셋을 사용할 수 있습니다. 훈련셋을 몇 번 반복해서 학습할 것인가를 정하는 것이 에포크(epochs)라고 했습니다. 초기에는 에포크가 증가될수록 검증셋의 평가 결과도 좋아집니다.

이 상태는 아직 학습이 덜 된 상태 즉 학습을 더 하면 성능이 높아질 가능성이 있는 상태입니다. 이를 `언더피팅(underfitting)`이라고 합니다. 담임선생님 입장에서 학생들을 평생 반복 학습만 시킬 수 없으므로 (하교도 해야하고, 퇴근도 해야하고) 학생들의 학습 상태를 보면서 ‘아직 학습이 덜 되었으니 계속 반복하도록!’ 또는 ‘충분히 학습했으니 그만해도 돼’ 라는 판단을 내려야 합니다. 그 판단 기준이 무엇일까요? 에포크를 계속 증가시키다보면 더 이상 검증셋의 평가는 높아지지 않고 오버피팅이 되어 오히려 틀린 개수가 많아집니다. 이 시점이 적정 반복 횟수로 보고 학습을 중단합니다. 이를 `조기종료(early stopping)`이라고 합니다.

`검증셋이 있다면 학습 중단 시점을 정할 수 있습니다.`

### cross-validation

validation set 을 변경해가며 검증하고, 각 결과의 평균 및 분산을 검토한다.

모의고사 5회로만 검증셋을 사용할 경우 여러 가지 문제가 발생할 수 있습니다.

- 모의고사 5회에서 출제가 되지 않는 분야가 있을 수 있습니다.
- 모의고사 5회가 작년 수능이나 올해 수능 문제와 많이 다를 수도 있습니다.
- 모의고사 5회가 모의고사 1회~4회와 난이도 및 범위가 다를 수도 있습니다.

이런 이유로 모의고사 5회로만 검증셋을 사용하기에는 객관적인 평가가 이루어졌다고 보기 힘듭니다. 이 때 사용하는 것이 교차검증(cross-validation) 입니다. 하는 방법은 다음과 같습니다.

- 모의고사 1회~4회를 학습한 뒤 모의고사 5회로 평가를 수행합니다.
- 학습된 상태를 초기화한 후 다시 모의고사 1, 2, 3, 5회를 학습한 뒤 4회로 검증합니다.
- 학습된 상태를 초기화한 후 다시 모의고사 1, 2, 4, 5회를 학습한 뒤 3회로 검증합니다.
- 학습된 상태를 초기화한 후 다시 모의고사 1, 3, 4, 5회를 학습한 뒤 2회로 검증합니다.
- 학습된 상태를 초기화한 후 다시 모의고사 2, 3, 4, 5회를 학습한 뒤 1회로 검증합니다.

다섯 번의 검증결과를 평균 내어 이 평균값으로 성능을 정의합니다. 검증결과의 분산도 의미가 있을 것 같습니다. 검증셋이 다르다고 해서 결과가 많이 차이나는 것보다 평균이 낮더라도 안정적인 결과를 내는 것이 더 좋은 모델일 수 있습니다.

[출처]([https://tykimos.github.io/2017/03/25/Dataset_and_Fit_Talk/](https://tykimos.github.io/2017/03/25/Dataset_and_Fit_Talk/))

## 딥러닝 네트워크 설계하기

```jsx
model=keras.models.Sequential()
model.add(keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(32, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

print('Model에 추가된 Layer 개수: ', len(model.layers))
```

이런 간단한 코드만으로도 숫자 손글씨를 인식해 낼 수 있다면***, IoT 농장에서 귤이 잘 익었는지 아닌지 판단한다거나, 사진 속 인물이 웃고 있는지 무표정인지 파악을 하는 것도 어렵지 않을 겁니다.***

Conv2D 레이어의 첫 번째 인자는 사용하는 이미지 특징의 수. (고려하는 이미지 특징의 수) ⇒ 단순한 형태라면 입력값이 작고, 만약 얼굴 사진같은 이미지라면 더 복잡하니까, 더 커야겠지!

Dense 레이어의 첫 번째 인자는 classifier 에 사용되는 뉴런의 수. 값이 클수록 더 복잡한 classifier. 만약 알파벳을 분류한다고 하면, 알파벳은 총 52개니까, 52보다 큰 64,128 등등

⇒ 이렇게 배수로 가는 이유가 있었던거같은데...

마지막 Dense 레이어는 결과적으로 분류해 내야 하는 클래스의 수, 알파벳의 경우 52겠지.

## 딥러닝 네트워크 학습시키기

### channel

(여기서 채널수 1은 흑백 이미지를 의미합니다. 컬러 이미지라면 R, G, B 세 가지 값이 있기 때문에 3이겠죠?)

각 이미지 픽셀 데이터를 구성하는 구분 값 수? 정도로 이해하면 될 것 같다

When you feed a CNN colored images, those images come in three channels: Red, Green, Blue.

Say we have a 32 x 32 image like in CIFAR-10. For each of the 32 x 32 pixels, there is a value for the red channel, the green, and the blue, (this value is likely different cross the channels). The CNN interprets one sample as 3 x 32 x 32 block.

In later layers of a CNN, you can have more than 3 channels, with some networks having 100+ channels. These channels function just like the RGB channels, but these channels are an abstract version of color, with each channel representing some aspect of information about the image.

[출처]([https://www.quora.com/What-do-channels-refer-to-in-a-convolutional-neural-network](https://www.quora.com/What-do-channels-refer-to-in-a-convolutional-neural-network))

```jsx
x_train_reshaped=x_train_norm.reshape( -1, 28, 28, 1) # 데이터갯수에 -1을 쓰면 reshape시 자동계산됩니다.
```

## 얼마나 잘 만들었는지 확인하기

model.evaluate() 대신 model.predict()를 사용하면 model이 입력값을 보고 실제로 추론한 확률분포를 출력할 수 있습니다. 우리가 만든 model이란 사실 10개의 숫자 중 어느 것일지에 대한 확률값을 출력하는 함수입니다.

이 함수의 출력값 즉 확률값이 가장 높은 숫자가 바로 model이 추론한 숫자가 되는 거죠.

ㅇㅎ...

즉, model.predict() 를 통해 얻어지는 벡터는 model이 추론한 결과가 각각 0, 1, 2, …, 7, 8, 9일 확률을 의미하는것!

틀린 경우를 살펴보면 model도 추론 결과에 대한 확신도가 낮고 매우 혼란스러워 한다는 것을 알 수 있습니다. ⇒ predict() 를 통해 얻어지는 벡터 값에 들어있는 확률들 간의 차이가 옳게 예측했을 때와 다르게 크지 않다!

model의 추론 결과를 시각화하여 살펴보는 것은 향후 model성능 개선에 도움이 되는 아이디어를 얻을 수 있는 좋은 방법 중 하나입니다.

## 더 좋은 네트워크 만들어 보기

딥러닝 네트워크의 구조는 바꾸지 않으면서, 더 좋은 네트워크를 만드는 방법??

하이퍼파라미터를 변경해보자!

- Conv2D 레이어의 input 값 변경( 고려하는 이미지 특징 수 변경 )
- Dense 레이어의 뉴련 수 변경 ⇒ 복잡도 상승 혹은 감소
- epoch 변경 ⇒ 더 많이 혹은 더 적게 학습

[변경 내역](https://www.notion.so/b0bbfabf3f5c4e6399adb02cc65b8707)

## 프로젝트

[https://github.com/bluecandle/2020_AIFFEL/tree/master/daily_notes/codes/e1_code](https://github.com/bluecandle/2020_AIFFEL/tree/master/daily_notes/codes/e1_code)