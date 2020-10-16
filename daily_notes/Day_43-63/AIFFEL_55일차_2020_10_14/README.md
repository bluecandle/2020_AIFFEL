# AIFFEL_55일차 2020.10.14

Tags: AIFFEL_DAILY

### 일정

![https://i.redd.it/ypjkwzv6b0k51.png](https://i.redd.it/ypjkwzv6b0k51.png)

- [x]  LMS E-19

# [E-19] 흐린 사진을 선명하게

![https://i.redd.it/ypjkwzv6b0k51.png](https://i.redd.it/ypjkwzv6b0k51.png)

이미지 생성형 기술이 효과적으로 사용되는 사례로 저해상도의 이미지를 고해상도 이미지로 변환하는 ***Super Resolution***

GAN 관련 기술이 일반적인 CNN 기술보다 훨씬 정밀한 고해상도 이미지를 생성해 내는데 효과적.

하지만 고화질의 이미지를 생성해 내는 GAN 모델을 학습하는 것은 오랜 시간이 소요되는 과정.

잘 학습된 모델을 활용한 고해상도 변환을 체험하는 데 보다 중점을 두어 진행

## **학습 목표**

---

- Super Resolution과 그 과정 수행에 필요한 기본 개념을 이해합니다.
- Super Resolution에 사용되는 대표적인 2개의 구조(SRCNN, SRGAN)에 대해 이해하고 활용합니다.

## Super Resolution(초해상화)이란??

---

[http://blog.lgdisplay.com/2014/03/모니터-핵심-디스플레이의-스펙-따라잡기-해상도/](http://blog.lgdisplay.com/2014/03/%eb%aa%a8%eb%8b%88%ed%84%b0-%ed%95%b5%ec%8b%ac-%eb%94%94%ec%8a%a4%ed%94%8c%eb%a0%88%ec%9d%b4%ec%9d%98-%ec%8a%a4%ed%8e%99-%eb%94%b0%eb%9d%bc%ec%9e%a1%ea%b8%b0-%ed%95%b4%ec%83%81%eb%8f%84/)

[https://blog.lgdisplay.com/2014/07/그림으로-쉽게-알아보는-hd-해상도의-차이/](https://blog.lgdisplay.com/2014/07/%EA%B7%B8%EB%A6%BC%EC%9C%BC%EB%A1%9C-%EC%89%BD%EA%B2%8C-%EC%95%8C%EC%95%84%EB%B3%B4%EB%8A%94-hd-%ED%95%B4%EC%83%81%EB%8F%84%EC%9D%98-%EC%B0%A8%EC%9D%B4/)

### 픽셀이란? 빛의 3원색을 혼합하여 색을 나타내는 방식은 무엇?

픽셀은 디스플레이를 구성하고 있는 가장 작은 단위

빛의 3원색인 적•녹•청을 혼합하여 색을 나타내는 RGB 방식에 있어서 RGB 각각을 Sub-Pixel이라고 하고, RGB 3개를 모아 놓은 것을 One-Pixel

### 픽셀과 해상도의 관계

픽셀의 개수가 많아질수록 그림은 더 선명하게 보이게 되는데요. 동일한 크기의 화면으로 비교했을 때, 정해진 공간에 더 많은 수의 픽셀이 들어간다면 그만큼 정밀하게 표현할 수 있는 원리

1 인치 당 픽셀이 몇 개인지를 나타내는 PPI(pixel per inch)는 픽셀 밀도를 나타내는데요. 예를 들어, 326PPI라는 말은 1 인치 당 326개의 픽셀이 들어있고, 1 in² (평방인치)에 106,272개 (326의 제곱)의 픽셀이 들어있다는 뜻입니다. 이는 곧 따져보면 1개의 픽셀 당 빛의 삼원색인 적, 녹, 청3가지 서브픽셀로 이루어져있기 때문에, 326의 3배인 978개의 픽셀이 있는 것과 마찬가지

## Super Resolution 활용 사례

---

1. cctv 저화질 영상 변형
2. 영상 복원

![AIFFEL_55%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8E%E1%85%A1%202020%2010%2014%2052a14b3061fe45f4ab004f1842dbf1a5/Untitled.png](AIFFEL_55%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8E%E1%85%A1%202020%2010%2014%2052a14b3061fe45f4ab004f1842dbf1a5/Untitled.png)

일반적으로 고해상도의 의료 영상을 얻는데 매우 긴 스캔 시간이 필요하거나, 환자의 몸에 많은 방사선이 노출되어 다른 부작용을 초래할 수 있습니다. Super Resolution 기술은 이러한 단점을 극복할 수 있는 기술 중 하나로 사용됩니다.

## Super Resolution을 어렵게 만드는 요인들

---

### ill-posed problem

저해상도 이미지에 대해 여러 개의 고해상도 이미지가 나올 수 있다는 것 : ill-posed (inverse) problem

![AIFFEL_55%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8E%E1%85%A1%202020%2010%2014%2052a14b3061fe45f4ab004f1842dbf1a5/Untitled%201.png](AIFFEL_55%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8E%E1%85%A1%202020%2010%2014%2052a14b3061fe45f4ab004f1842dbf1a5/Untitled%201.png)

일반적으로 Super Resolution 모델을 학습시키기 위한 데이터를 구성하는 과정

1. 먼저 고해상도 이미지를 준비하고 특정한 처리과정을 거쳐 저해상도 이미지를 생성
2. 생성된 저해상도 이미지를 입력으로 원래의 고해상도 이미지를 복원하도록 학습

### Super Resolution 문제의 복잡도

![AIFFEL_55%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8E%E1%85%A1%202020%2010%2014%2052a14b3061fe45f4ab004f1842dbf1a5/Untitled%202.png](AIFFEL_55%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8E%E1%85%A1%202020%2010%2014%2052a14b3061fe45f4ab004f1842dbf1a5/Untitled%202.png)

녹색으로 나타난 2x2 이미지 픽셀을 입력으로 3x3 크기의 이미지 만드는 경우 새롭게 생성해야 하는 정보는 최소 5개 픽셀(회색)이며, 4x4의 경우 12개, 5x5의 경우 21개의 정보를 생성해야 함.

원래 가진 제한된 정보(녹색 픽셀)만을 이용해 많은 정보(회색 픽셀)를 만들어내는 과정은 매우 복잡하며 그만큼 잘못된 정보를 만들어 낼 가능성 또한 높다.

### 결과를 평가하는 데 있어 흔히 사용되는 정량적 평가 척도와 사람이 시각적으로 관찰하여 내린 평가가 잘 일치하지 않는다는 것

![AIFFEL_55%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8E%E1%85%A1%202020%2010%2014%2052a14b3061fe45f4ab004f1842dbf1a5/Untitled%203.png](AIFFEL_55%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8E%E1%85%A1%202020%2010%2014%2052a14b3061fe45f4ab004f1842dbf1a5/Untitled%203.png)

결과 2의 이미지에 세밀한 정보가 잘 표현되었다고 생각했지만 실제 평가 결과는 결과 1 이미지에 쓰인 숫자가 더 높은 것을 확인할 수 있습니다.

## 가장 쉬운 Super Resolution

---

### Interpoloation

[https://bskyvision.com/789](https://bskyvision.com/789)

보간법이란 알려진 값을 가진 두 점 사이 어느 지점의 값이 얼마일지를 추정하는 기법

![AIFFEL_55%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8E%E1%85%A1%202020%2010%2014%2052a14b3061fe45f4ab004f1842dbf1a5/Untitled%204.png](AIFFEL_55%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8E%E1%85%A1%202020%2010%2014%2052a14b3061fe45f4ab004f1842dbf1a5/Untitled%204.png)

a와 b사이에 위치한 x의 값, 즉 f(x)가 무엇인지를 추정하는 것이 바로 보간법

**선형 보간법**

두 점 사이에 직선을 그립니다. 그리고 그 선을 이용해서 f(x)를 추정

![AIFFEL_55%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8E%E1%85%A1%202020%2010%2014%2052a14b3061fe45f4ab004f1842dbf1a5/Untitled%205.png](AIFFEL_55%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8E%E1%85%A1%202020%2010%2014%2052a14b3061fe45f4ab004f1842dbf1a5/Untitled%205.png)

**삼차보간법(cubic interpolation)**

![AIFFEL_55%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8E%E1%85%A1%202020%2010%2014%2052a14b3061fe45f4ab004f1842dbf1a5/Untitled%206.png](AIFFEL_55%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8E%E1%85%A1%202020%2010%2014%2052a14b3061fe45f4ab004f1842dbf1a5/Untitled%206.png)

3차 함수를 이용해서 미지의 값을 추정하면 일차 함수를 통해 찾는 것보다 훨씬 더 부드러운 결과.

**쌍선형보간법(bilinear interpolation)과 쌍삼차보간법(bicubic interpolation)**

선형보간법을 2차원으로 확장시킨 것이 바로 쌍선형보간법이고, 삼차보간법을 2차원으로 확장시킨 것이 쌍삼차보간법입니다. 2차원으로 확장시킨 것 외에 원리상 달라진 것은 없습니다. 쌍선형보간법은 이웃한 4(=2x2)개의 점을 참조하고, 쌍삼차보간법은 16(=4x4)개의 점을 참조.

```python
from skimage import data
import matplotlib.pyplot as plt

hr_image = data.chelsea() # skimage에서 제공하는 예제 이미지를 불러옵니다.
hr_shape = hr_image.shape[:2]

print(hr_image.shape) # 이미지의 크기를 출력합니다.
# => (300, 451, 3)
plt.figure(figsize=(6,3))
plt.imshow(hr_image)
```

크기를 줄여보자

```python
import cv2
lr_image = cv2.resize(hr_image, dsize=(150,100)) # (가로 픽셀 수, 세로 픽셀 수)

print(lr_image.shape)

plt.figure(figsize=(3,1))
plt.imshow(lr_image)
```

작아진 이미지를 interpolation을 통해 Super Resolution 해보기.

```python
bilinear_image = cv2.resize(
    lr_image, 
    dsize=(451, 300), # (가로 픽셀 수, 세로 픽셀 수) 
    interpolation=cv2.INTER_LINEAR # bilinear interpolation 적용
)

bicubic_image = cv2.resize(
    lr_image, 
    dsize=(451, 300), # (가로 픽셀 수, 세로 픽셀 수)
    interpolation=cv2.INTER_CUBIC # bicubic interpolation 적용
)

images = [bilinear_image, bicubic_image, hr_image]
titles = ["Bilinear", "Bicubic", "HR"]

plt.figure(figsize=(16,3))
for i, (image, title) in enumerate(zip(images, titles)):
    plt.subplot(1,3,i+1)
    plt.imshow(image)
    plt.title(title, fontsize=20)
```

![AIFFEL_55%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8E%E1%85%A1%202020%2010%2014%2052a14b3061fe45f4ab004f1842dbf1a5/Untitled%207.png](AIFFEL_55%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8E%E1%85%A1%202020%2010%2014%2052a14b3061fe45f4ab004f1842dbf1a5/Untitled%207.png)

⇒ 맨 오른쪽 고해상도 이미지에 비해 1,2번 이미지는 수염,털이 조금 흐릿하게 표현된다\\

특정 부분을 잘라내보자.

```python
# 특정 영역을 잘라낼 함수를 정의합니다.
def crop(image, left_top, x=50, y=100):
    return image[left_top[0]:(left_top[0]+x), left_top[1]:(left_top[1]+y), :]

# 잘라낼 영역의 좌표를 정의합니다.
left_tops = [(220,200)] *3 + [(90,120)] *3 + [(30,200)] *3

plt.figure(figsize=(16,10))
for i, (image, left_top, title) in enumerate(zip(images*3, left_tops, titles*3)):
    plt.subplot(3,3,i+1)
    plt.imshow(crop(image, left_top))
    plt.title(title, fontsize=20)
```

![AIFFEL_55%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8E%E1%85%A1%202020%2010%2014%2052a14b3061fe45f4ab004f1842dbf1a5/Untitled%208.png](AIFFEL_55%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8E%E1%85%A1%202020%2010%2014%2052a14b3061fe45f4ab004f1842dbf1a5/Untitled%208.png)

⇒ 확실히 차이가 있다! 그리고 bicubic interpolation 이 그나마 조금 더 나은 느낌.

## Deep Learning을 이용한 Super Resolution (1) SRCNN

---

지금 이 연구를 보면 너무 간단해 보이지만, Super Resolution 문제에 가장 처음 딥러닝을 적용한 연구로써 이후 많은 딥러닝 기반의 Super Resolution 연구에 큰 영향을 준 정말 멋진 작품.

![AIFFEL_55%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8E%E1%85%A1%202020%2010%2014%2052a14b3061fe45f4ab004f1842dbf1a5/Untitled%209.png](AIFFEL_55%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8E%E1%85%A1%202020%2010%2014%2052a14b3061fe45f4ab004f1842dbf1a5/Untitled%209.png)

1. 해상도 이미지(그림의 LR)를 bicubic interpolation 하여 원하는 크기로 이미지를 늘립니다.
2. SRCNN은 이 이미지(그림의 ILR)를 입력으로 사용.
3. 이 후 3개의 convolutional layer를 거쳐 고해상도 이미지를 생성
4. 생성된 고해상도 이미지와 실제 고해상도 이미지 사이의 차이를 역전파 하여 신경망의 가중치를 학습.

[https://d-tail.tistory.com/6](https://d-tail.tistory.com/6)

Patch extraction and representation : 저해상도 이미지에서 patch 추출

Non-linear mapping : 다차원의 patch들을 non-linear하게 다른 다차원의 patch들로 매핑

Reconstruction : 다차원 patch들로부터 고해상도 이미지 복원.

손실 함수는 MSE 사용.

## Deep Learning을 이용한 Super Resolution (2) SRCNN 이후 제안된 구조들

---

### VDSR(Very Deep Super Resolution)

![AIFFEL_55%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8E%E1%85%A1%202020%2010%2014%2052a14b3061fe45f4ab004f1842dbf1a5/Untitled%2010.png](AIFFEL_55%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8E%E1%85%A1%202020%2010%2014%2052a14b3061fe45f4ab004f1842dbf1a5/Untitled%2010.png)

20개의 convolutional layer를 사용

종 고해상도 이미지 생성 직전에 처음 입력 이미지를 더하는 residual learning을 이용.

### RDN(Residual Dense Network)

![AIFFEL_55%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8E%E1%85%A1%202020%2010%2014%2052a14b3061fe45f4ab004f1842dbf1a5/Untitled%2011.png](AIFFEL_55%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8E%E1%85%A1%202020%2010%2014%2052a14b3061fe45f4ab004f1842dbf1a5/Untitled%2011.png)

각각의 convolution layer 출력 결과로 생성된 특징들이 화살표를 따라 이 후 연산에서 여러 번 재활용

### RCAN (Residual Channel Attention Networks)

![AIFFEL_55%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8E%E1%85%A1%202020%2010%2014%2052a14b3061fe45f4ab004f1842dbf1a5/Untitled%2012.png](AIFFEL_55%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8E%E1%85%A1%202020%2010%2014%2052a14b3061fe45f4ab004f1842dbf1a5/Untitled%2012.png)

convolutional layer의 결과인 각각의 특징 맵을 대상으로 채널 간의 모든 정보가 균일한 중요도를 갖는 것이 아니라 **일부 중요한 채널에만 선택적으로 집중(Attention)**하도록 유도.

## SRCNN을 이용해 Super Resolution 도전하기

---

DIV2K 데이터셋은 많은 Super Resolution 연구에서 학습 및 평가에 사용되는 데이터셋이며 800개의 학습용 데이터셋 및 100개의 검증용 데이터셋으로 구성

`div2k/bicubic_x4`

DIV2K 데이터 셋 중에서 실제 고해상도 이미지를 대상으로 bicubic interpolation을 이용해 가로 및 세로 픽셀 수를 1/4배로 줄인 데이터셋.

저해상도 이미지와 원래 고해상도 이미지가 서로 한 쌍으로 구성.

SRCNN의 구현 및 학습 과정을 매우 간단하게 해보자 ( 실제 논문의 구현과는 차이가 있음 )

## 데이터 준비

해상도 이미지를 bicubic interpolation하여 고해상도 이미지와 동일한 크기로 만들기

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

# 데이터를 불러옵니다.
train, valid = tfds.load(
    "div2k/bicubic_x4", 
    split=["train","validation"],
    as_supervised=True
)

# 시각화를 위해 한 개의 데이터만 선택합니다.
for i, (lr, hr) in enumerate(valid):
    if i == 6: break
    
# 저해상도 이미지를 고해상도 이미지 크기로 bicubic interpolation 합니다.  
hr, lr = np.array(hr), np.array(lr)
bicubic_hr = cv2.resize(
    lr, 
    dsize=(hr.shape[1], hr.shape[0]), # 고해상도 이미지 크기로 설정
    interpolation=cv2.INTER_CUBIC # bicubic 설정
)

# 저해상도 및 고해상도 이미지를 시각화 합니다.
plt.figure(figsize=(20,10))
plt.subplot(1,2,1); plt.imshow(bicubic_hr)
plt.subplot(1,2,2); plt.imshow(hr)
```

특정부분 확대

```python
# 이미지의 특정 부분을 잘라내는 함수를 정의합니다.
def crop(image, left_top, x=200, y=200):
    return image[left_top[0]:(left_top[0]+x), left_top[1]:(left_top[1]+y), :]

# interpolation된 이미지와 고해상도 이미지의 동일한 부분을 각각 잘라냅니다.
left_top = (400, 500)
crop_bicubic_hr = crop(bicubic_hr, left_top)
crop_hr = crop(hr, left_top)

# 잘라낸 부분을 시각화 합니다.
plt.figure(figsize=(15,25))
plt.subplot(1,2,1); plt.imshow(crop_bicubic_hr); plt.title("Bicubic", fontsize=30)
plt.subplot(1,2,2); plt.imshow(crop_hr); plt.title("HR", fontsize=30)
```

![AIFFEL_55%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8E%E1%85%A1%202020%2010%2014%2052a14b3061fe45f4ab004f1842dbf1a5/Untitled%2013.png](AIFFEL_55%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8E%E1%85%A1%202020%2010%2014%2052a14b3061fe45f4ab004f1842dbf1a5/Untitled%2013.png)

⇒확대해보니 차이가 좀 있다!

SRCNN의 입력은 저해상도 이미지를 그대로 사용하는 것이 아닌, 만들고자 하는 고해상도 이미지 크기에 맞게 interpolation이 적용된 이미지.

사용할 DIV2K 데이터셋 내의 개별 이미지 크기가 크므로, 일부 영역을 임의로 잘라내어 학습에 활용

```python
import tensorflow as tf

def preprocessing(lr, hr):
    # 이미지의 크기가 크므로 (96,96,3) 크기로 임의 영역을 잘라내어 사용합니다.
    hr = tf.image.random_crop(hr, size=[96, 96, 3])
    hr = tf.cast(hr, tf.float32) / 255.
    
    # 잘라낸 고해상도 이미지의 가로, 세로 픽셀 수를 1/4배로 줄였다가
    # interpolation을 이용해 다시 원래 크기로 되돌립니다.
    # 이렇게 만든 저해상도 이미지를 입력으로 사용합니다.
    lr = tf.image.resize(hr, [96//4, 96//4], "bicubic")
    lr = tf.image.resize(lr, [96, 96], "bicubic")
    return lr, hr

train = train.map(preprocessing).shuffle(buffer_size=10).batch(16)
valid = valid.map(preprocessing).batch(16)
print("✅")
```

### SRCNN 구현

아주 간단한 형태! 논문 내용과는 세부사항에서 차이가 있다.

```python
from tensorflow.keras import layers, Sequential

# 3개의 convolutional layer를 갖는 Sequential 모델을 구성합니다.
srcnn = Sequential()
# 9x9 크기의 필터를 128개 사용합니다.
srcnn.add(layers.Conv2D(128, 9, padding="same", input_shape=(None, None, 3)))
srcnn.add(layers.ReLU())
# 5x5 크기의 필터를 64개 사용합니다.
srcnn.add(layers.Conv2D(64, 5, padding="same"))
srcnn.add(layers.ReLU())
# 5x5 크기의 필터를 64개 사용합니다.
srcnn.add(layers.Conv2D(3, 5, padding="same"))

srcnn.summary()
```

### SRCNN 학습

```python
srcnn.compile(
    optimizer="adam", 
    loss="mse"
)

srcnn.fit(train, validation_data=valid, epochs=1)
```

### SRCNN 테스트

SRCNN의 학습에는 꽤나 오랜시간이 소요.

학습이 완료된 SRCNN 모델을 준비

```python
import tensorflow as tf
import os

model_file = os.getenv('HOME')+'/aiffel/super_resolution/srcnn.h5'
srcnn = tf.keras.models.load_model(model_file)
```

저해상도 이미지를 입력받아 SRCNN을 사용하는 함수를 간단하게 정의하고, 이 함수를 이용해 SRCNN의 결과인 고해상도 이미지를 얻어봅시다.

```python
def apply_srcnn(image):
    sr = srcnn.predict(image[np.newaxis, ...]/255.)
    sr[sr > 1] = 1
    sr[sr < 0] = 0
    sr *= 255.
    return np.array(sr[0].astype(np.uint8))

srcnn_hr = apply_srcnn(bicubic_hr)
```

일부 영역을 잘라내어 시각적으로 비교

```python
# 자세히 시각화 하기 위해 3개 영역을 잘라냅니다.
# 아래는 잘라낸 부분의 좌상단 좌표 3개 입니다.
left_tops = [(400,500), (300,1200), (0,1000)]

images = []
for left_top in left_tops:
    img1 = crop(bicubic_hr, left_top, 200, 200)
    img2 = crop(srcnn_hr , left_top, 200, 200)
    img3 = crop(hr, left_top, 200, 200)
    images.extend([img1, img2, img3])

labels = ["Bicubic", "SRCNN", "HR"] * 3

plt.figure(figsize=(18,18))
for i in range(9):
    plt.subplot(3,3,i+1) 
    plt.imshow(images[i])
    plt.title(labels[i], fontsize=30)
```

![AIFFEL_55%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8E%E1%85%A1%202020%2010%2014%2052a14b3061fe45f4ab004f1842dbf1a5/Untitled%2014.png](AIFFEL_55%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8E%E1%85%A1%202020%2010%2014%2052a14b3061fe45f4ab004f1842dbf1a5/Untitled%2014.png)

⇒ 시각화 결과 bicubic interpolation 결과보다 조금 더 선명해 졌지만 원래 고해상도 이미지에 비해 만족할만한 성능은 아닌 것 같습니다. DIV2K 데이터셋이 비교적 세밀한 구조의 이미지가 많아 SRCNN과 같이 간단한 구조로는 더 이상 학습되지 않는 것으로 보입니다.

⇒ 비교적 간단한 구조의 이미지에 대해서는 꽤나 만족할 만한 성능을 보여줍니다.

## Deep Learning을 이용한 Super Resolution (3) SRGAN

---

GAN(Generative Adversarial Networks) 을 활용한 Super Resolution 과정

고해상도 이미지를 만들어 내는 데 GAN이 어떻게 활용될 수 있고, 이를 활용했을 때 어떠한 장점이 있는지

[https://www.samsungsds.com/kr/insights/Generative-adversarial-network-AI.html?referrer=https://aiffelstaticprd.blob.core.windows.net/](https://www.samsungsds.com/kr/insights/Generative-adversarial-network-AI.html?referrer=https://aiffelstaticprd.blob.core.windows.net/)

비지도학습 GAN은 원 데이터가 가지고 있는 확률분포를 추정하도록 하고, 인공신경망이 그 분포를 만들어 낼 수 있도록 한다는 점에서 단순한 군집화 기반의 비지도학습과 차이가 있습니다.

GAN에서 다루고자 하는 모든 데이터는 확률분포를 가지고 있는 랜덤변수(Random Variable)

**랜덤변수는 측정할 때마다 다른 값이 나옵니다. 하지만, 특정한 확률분포를 따르는 숫자를 생성하므로, 랜덤변수에 대한 확률분포를 안다는 이야기는 랜덤변수 즉 데이터에 대한 전부를 이해하고 있다는 것과 같습니다.**

확률분포를 알면 그 데이터의 예측 기댓값, 데이터의 분산을 즉각 알아낼 수 있어 데이터의 통계적 특성을 바로 분석할 수 있으며, 주어진 확률분포를 따르도록 데이터를 임의 생성하면 그 데이터는 확률분포를 구할 때 사용한 원 데이터와 유사한 값을 갖는다.

⇒ 즉, GAN과 같은 비지도학습이 가능한 머신러닝 알고리즘으로 **데이터에 대한 확률분포를 모델링** 할 수 있게 되면, **원 데이터와 확률분포를 정확히 공유하는 무한히 많은 새로운 데이터를 새로 생성할 수 있음.**

수학적으로 생성자 G는 앞에서 말한 원 데이터의 확률분포를 알아내려고 노력하며, 학습이 종료된 후에는 원 **데이터의 확률분포를 따르는 새로운 데이터**를 만들어 내게 됩니다.

![AIFFEL_55%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8E%E1%85%A1%202020%2010%2014%2052a14b3061fe45f4ab004f1842dbf1a5/Untitled%2015.png](AIFFEL_55%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8E%E1%85%A1%202020%2010%2014%2052a14b3061fe45f4ab004f1842dbf1a5/Untitled%2015.png)

파란색 점선인 분류자 D는 더 이상 분류를 해도 의미가 없는 0.5라는 확률 값을 뱉어내게 되죠. 이것은 동전을 던져서 앞면을 진실, 뒷면을 거짓이라고 했을 때, 진실을 맞출 확률이 0.5가 되는 것처럼 GAN에 의해 만들어진 데이터가 진짜 인지 가짜인지 맞출 확률이 0.5가 되면서 분류자가 의미 없게 되는 겁니다.

이론적으로 생성자 G가 실제 데이터와 거의 유사한 데이터를 만들어 낼 수 있는 상황이 되었음을 의미하죠.

[https://www.samsungsds.com/kr/insights/Generative-adversarial-network-AI-2.html?referrer=https://aiffelstaticprd.blob.core.windows.net/](https://www.samsungsds.com/kr/insights/Generative-adversarial-network-AI-2.html?referrer=https://aiffelstaticprd.blob.core.windows.net/)

***“What I cannot create, I do not understand.”***

정답 없이 새로운 것을 지속적으로 생성해 낼 수 있는 능력을 가진다는 것은 그 데이터를 완전히 이해하고 있다는 의미로서 생성할 수 있는 능력을 가지게 되면, 분류하는 것도 자동적으로 쉬워진다는 이야기.

*어려운 수학 문제를 혼자서 쉽게 풀 수 있어도, 다른 사람에게 이해시키기 위해 설명하는 것은 그보다 어렵지만 다른 사람에게 설명을 잘 할 수 있다면, 그 문제를 푸는 일은 식은 죽 먹기일 것*

분류 모델을 먼저 학습시킨 후, 생성 모델을 학습시키는 과정을 서로 주고받으면서 반복.

분류 모델의 학습은 크게 두 가지 단계로 이루어져 있습니다. 하나는 진짜 데이터를 입력해서 네트워크가 해당 데이터를 진짜로 분류하도록 학습시키는 과정이고 두 번째는 첫 번째와 반대로 생성 모델에서 생성한 가짜 데이터를 입력해서 해당 데이터를 가짜로 분류하도록 학습하는 과정.

이 과정을 통해 분류 모델은 진짜 데이터를 진짜로, 가짜 데이터를 가짜로 분류할 수 있게 됩니다. 분류 모델을 학습시킨 다음에는 학습된 분류 모델을 속이는 방향으로 생성 모델을 학습.

생성 모델에서 만들어낸 가짜 데이터를 판별 모델에 입력하고, 가짜 데이터를 진짜라고 분류할 만큼 진짜 데이터와 유사한 데이터를 만들어 내도록 생성 모델을 학습.

### SRGAN = Super Resoultion + GAN

Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network

![AIFFEL_55%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8E%E1%85%A1%202020%2010%2014%2052a14b3061fe45f4ab004f1842dbf1a5/Untitled%2016.png](AIFFEL_55%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8E%E1%85%A1%202020%2010%2014%2052a14b3061fe45f4ab004f1842dbf1a5/Untitled%2016.png)

SRGAN에서 사용하는 loss function

![AIFFEL_55%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8E%E1%85%A1%202020%2010%2014%2052a14b3061fe45f4ab004f1842dbf1a5/Untitled%2017.png](AIFFEL_55%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8E%E1%85%A1%202020%2010%2014%2052a14b3061fe45f4ab004f1842dbf1a5/Untitled%2017.png)

조금 특별한 부분은 아래 그림으로 나타낸 **content loss** 부분

![AIFFEL_55%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8E%E1%85%A1%202020%2010%2014%2052a14b3061fe45f4ab004f1842dbf1a5/Untitled%2018.png](AIFFEL_55%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8E%E1%85%A1%202020%2010%2014%2052a14b3061fe45f4ab004f1842dbf1a5/Untitled%2018.png)

content loss는 Generator를 이용해 얻어낸 (가짜) 고해상도 이미지를 실제 (진짜) 고해상도 이미지와 직접 비교하는 것이 아니라, 각 이미지를 이미지넷으로 사전 학습된(pre-trained) VGG 모델에 입력하여 나오는 feature map에서의 차이를 계산.

***즉, 이전에 학습했던 SRCNN은 생성해낸 고해상도 이미지를 원래 고해상도 이미지와 직접 비교하여 loss를 계산했지만, SRGAN에서는 생성된 고해상도 이미지와 실제 고해상도 이미지를 VGG에 입력하여 모델 중간에서 추출해낸 특징을 비교해서 loss를 계산***

SRGAN은 VGG를 이용한 **content loss** 및 GAN을 사용함으로써 발생하는 **adversarial loss**를 합하여 최종적으로 **perceptual loss**라고 정의하며 이를 학습에 이용

이렇게 복잡한 손실 함수를 정의한 이유가 무엇일까요? 장점에 대해 생각해보자.

## SRGAN을 이용해 Super Resolution 도전하기

⇒ 내일 이어서

프로젝트 결과물 : [https://github.com/bluecandle/2020_AIFFEL/blob/master/daily_notes/exploration_codes/e19_code/E19.ipynb](https://github.com/bluecandle/2020_AIFFEL/blob/master/daily_notes/exploration_codes/e19_code/E19.ipynb)