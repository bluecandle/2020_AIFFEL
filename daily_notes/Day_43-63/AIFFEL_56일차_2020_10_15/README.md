# AIFFEL_56일차 2020.10.15

Tags: AIFFEL_DAILY

### 일정

![https://i.redd.it/ypjkwzv6b0k51.png](https://i.redd.it/ypjkwzv6b0k51.png)

- [x]  lms E-19
- [x]  GD Proj2

# [E-19] 흐린 사진을 선명하게

![https://i.redd.it/ypjkwzv6b0k51.png](https://i.redd.it/ypjkwzv6b0k51.png)

## SRGAN을 이용해 Super Resolution 도전하기

데이터 준비 : SRCNN은 저해상도 이미지를 interpolation 하여 고해상도로 변환 후 입력으로 사용, SRGAN은 그렇게 하지 않는다.

```python
import tensorflow as tf
train, valid = tfds.load(
    "div2k/bicubic_x4", 
    split=["train","validation"],
    as_supervised=True
)
def preprocessing(lr, hr):
    hr = tf.cast(hr, tf.float32) /255.
        
    # 이미지의 크기가 크므로 (96,96,3) 크기로 임의 영역을 잘라내어 사용합니다.
    hr_patch = tf.image.random_crop(hr, size=[96,96,3])
        
    # 잘라낸 고해상도 이미지의 가로, 세로 픽셀 수를 1/4배로 줄입니다
    # 이렇게 만든 저해상도 이미지를 SRGAN의 입력으로 사용합니다.
    lr_patch = tf.image.resize(hr_patch, [96//4, 96//4], "bicubic")
    return lr_patch, hr_patch

train = train.map(preprocessing).shuffle(buffer_size=10).repeat().batch(8)
valid = valid.map(preprocessing).repeat().batch(8)
print("✅")
```

### SRGAN 구현하기

![README/Untitled.png](README/Untitled.png)

`k9n64s1`라는 표기는 Convolutional layer 내의 hyperparameter 설정에 대한 정보이며, k는 kernel size, n은 사용 필터의 수, s는 stride를 나타냅니다.

⇒ 오호... 이렇게 표현할 수도 있는거구나!

***Generator 구현***

```python
from tensorflow.keras import Input, Model

# 그림의 파란색 블록을 정의합니다.
def gene_base_block(x):
    out = layers.Conv2D(64, 3, 1, "same")(x)
    out = layers.BatchNormalization()(out)
    out = layers.PReLU(shared_axes=[1,2])(out)
    out = layers.Conv2D(64, 3, 1, "same")(out)
    out = layers.BatchNormalization()(out)
    return layers.Add()([x, out]) # Skip Connection

# 그림의 뒤쪽 연두색 블록을 정의합니다.
def upsample_block(x):
    out = layers.Conv2D(256, 3, 1, "same")(x)
    # 그림의 PixelShuffler 라고 쓰여진 부분을 아래와 같이 구현합니다.
    out = layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2))(out)
    return layers.PReLU(shared_axes=[1,2])(out)
    
# 전체 Generator를 정의합니다.
def get_generator(input_shape=(None, None, 3)):
    inputs = Input(input_shape)
    
    out = layers.Conv2D(64, 9, 1, "same")(inputs)
    out = residual = layers.PReLU(shared_axes=[1,2])(out)
    
    for _ in range(5):
        out = gene_base_block(out)
    
    out = layers.Conv2D(64, 3, 1, "same")(out)
    out = layers.BatchNormalization()(out)
    out = layers.Add()([residual, out])
    
    for _ in range(2):
        out = upsample_block(out)
        
    out = layers.Conv2D(3, 9, 1, "same", activation="tanh")(out)
    return Model(inputs, out)

print("✅")
```

***Discriminator 구현***

![README/Untitled%201.png](README/Untitled%201.png)

```python
# 그림의 파란색 블록을 정의합니다.
def disc_base_block(x, n_filters=128):
    out = layers.Conv2D(n_filters, 3, 1, "same")(x)
    out = layers.BatchNormalization()(out)
    out = layers.LeakyReLU()(out)
    out = layers.Conv2D(n_filters, 3, 2, "same")(out)
    out = layers.BatchNormalization()(out)
    return layers.LeakyReLU()(out)

# 전체 Discriminator 정의합니다.
def get_discriminator(input_shape=(None, None, 3)):
    inputs = Input(input_shape)
    
    out = layers.Conv2D(64, 3, 1, "same")(inputs)
    out = layers.LeakyReLU()(out)
    out = layers.Conv2D(64, 3, 2, "same")(out)
    out = layers.BatchNormalization()(out)
    out = layers.LeakyReLU()(out)
    
    for n_filters in [128, 256, 512]:
        out = disc_base_block(out, n_filters)
    
    out = layers.Dense(1024)(out)
    out = layers.LeakyReLU()(out)
    out = layers.Dense(1, activation="sigmoid")(out)
    return Model(inputs, out)

print("✅")
```

VGG19 을 이용하여 content loss 를 계산하기 때문에, pre-trained VGG19 을 가져와보자.

```python
from tensorflow.python.keras import applications
def get_feature_extractor(input_shape=(None, None, 3)):
    vgg = applications.vgg19.VGG19(
        include_top=False, 
        weights="imagenet", 
        input_shape=input_shape
    )
    # 아래 vgg.layers[20]은 vgg 내의 마지막 convolutional layer 입니다.
    return Model(vgg.input, vgg.layers[20].output)

print("✅")
```

### SRGAN 학습하기

시간 오래걸리니까 200번만 해보자.

```python
from tensorflow.keras import layers,losses, metrics, optimizers
generator = get_generator()
discriminator = get_discriminator()
vgg = get_feature_extractor()

# 사용할 loss function 및 optimizer 를 정의합니다.
bce = losses.BinaryCrossentropy(from_logits=False)
mse = losses.MeanSquaredError()
gene_opt = optimizers.Adam()
disc_opt = optimizers.Adam()

def get_gene_loss(fake_out):
    return bce(tf.ones_like(fake_out), fake_out)

def get_disc_loss(real_out, fake_out):
    return bce(tf.ones_like(real_out), real_out) + bce(tf.zeros_like(fake_out), fake_out)

@tf.function
def get_content_loss(hr_real, hr_fake):
    hr_real = applications.vgg19.preprocess_input(hr_real)
    hr_fake = applications.vgg19.preprocess_input(hr_fake)
    
    hr_real_feature = vgg(hr_real) / 12.75
    hr_fake_feature = vgg(hr_fake) / 12.75
    return mse(hr_real_feature, hr_fake_feature)

@tf.function
def step(lr, hr_real):
    with tf.GradientTape() as gene_tape, tf.GradientTape() as disc_tape:
        hr_fake = generator(lr, training=True)
        
        real_out = discriminator(hr_real, training=True)
        fake_out = discriminator(hr_fake, training=True)
        
        perceptual_loss = get_content_loss(hr_real, hr_fake) + 1e-3 * get_gene_loss(fake_out)
        discriminator_loss = get_disc_loss(real_out, fake_out)
        
    gene_gradient = gene_tape.gradient(perceptual_loss, generator.trainable_variables)
    disc_gradient = disc_tape.gradient(discriminator_loss, discriminator.trainable_variables)
    
    gene_opt.apply_gradients(zip(gene_gradient, generator.trainable_variables))
    disc_opt.apply_gradients(zip(disc_gradient, discriminator.trainable_variables))
    return perceptual_loss, discriminator_loss

gene_losses = metrics.Mean()
disc_losses = metrics.Mean()

for epoch in range(1, 2):
    for i, (lr, hr) in enumerate(train):
        g_loss, d_loss = step(lr, hr)
        
        gene_losses.update_state(g_loss)
        disc_losses.update_state(d_loss)
        
        # 10회 반복마다 loss를 출력합니다.
        if (i+1) % 10 == 0:
            print(f"EPOCH[{epoch}] - STEP[{i+1}] \nGenerator_loss:{gene_losses.result():.4f} \nDiscriminator_loss:{disc_losses.result():.4f}", end="\n\n")
        
        if (i+1) == 200:
            break
            
    gene_losses.reset_states()
    disc_losses.reset_states()
```

### SRGAN 테스트

pre-trained 모델을 들고와서 테스트해보자.

SRGAN은 크게 두 개의 신경망(Generator, Discriminator)으로 구성되어 있지만, 테스트에는 저해상도 입력을 넣어 고해상도 이미지를 출력하는 Generator만 이용,

학습이 완료된 Generator 를 불러오자

```python
$ wget https://aiffelstaticprd.blob.core.windows.net/media/documents/srgan_G.h5
$ mv srgan_G.h5 ~/aiffel/super_resolution
```

```python
import tensorflow as tf
import os

model_file = os.getenv('HOME')+'/aiffel/super_resolution/srgan_G.h5'
srgan = tf.keras.models.load_model(model_file)
```

테스트 진행하는 함수 정의

```python
def apply_srgan(image):
    image = tf.cast(image[np.newaxis, ...], tf.float32)
    sr = srgan.predict(image)
    sr = tf.clip_by_value(sr, 0, 255)
    sr = tf.round(sr)
    sr = tf.cast(sr, tf.uint8)
    return np.array(sr)[0]

train, valid = tfds.load(
    "div2k/bicubic_x4", 
    split=["train","validation"],
    as_supervised=True
)

for i, (lr, hr) in enumerate(valid):
    if i == 6: break

srgan_hr = apply_srgan(lr)
print("✅")
```

일부 영역을 잘라내어 시각적으로 비교

```python
# 자세히 시각화 하기 위해 3개 영역을 잘라냅니다.
# 아래는 잘라낸 부분의 좌상단 좌표 3개 입니다.
left_tops = [(400,500), (300,1200), (0,1000)]

images = []
for left_top in left_tops:
    img1 = crop(bicubic_hr, left_top, 200, 200)
    img2 = crop(srgan_hr , left_top, 200, 200)
    img3 = crop(hr, left_top, 200, 200)
    images.extend([img1, img2, img3])

labels = ["Bicubic", "SRGAN", "HR"] * 3

plt.figure(figsize=(18,18))
for i in range(9):
    plt.subplot(3,3,i+1) 
    plt.imshow(images[i])
    plt.title(labels[i], fontsize=30)
```

![README/Untitled%202.png](README/Untitled%202.png)

⇒ Bicubic 보다 훨씬 HR 이미지에 가까운 결과물이 나왔다!

![README/Untitled%203.png](README/Untitled%203.png)

⇒ 세부 실험결과

SRResNet : SRGAN의 Generator

(Generator 구조만 이용해 SRCNN과 비슷하게 MSE 손실함수로 학습한 결과)

오른쪽으로 갈 수록 GAN 및 VGG 구조를 이용하여 점점 더 이미지 내 세부적인 구조가 선명해짐을 알 수 있습니다.

## Super Resolution 결과 평가하기

---

정량적인 평가 척도 몇 가지 존재.

### PSNR과 SSIM

---

**PSNR(Peak Signal-to-Noise Ratio)**: 영상 내 신호가 가질 수 있는 최대 신호에 대한 잡음(noise)의 비율.

일반적으로 영상을 압축했을 때 화질이 얼마나 손실되었는지 평가하는 목적.

데시벨(db) 단위를 사용하며, PSNR 수치가 높을 수록 원본 영상에 비해 손실이 적다는 의미

**SSIM(Structural Similarity Index Map)**: 영상의 구조 정보를 고려하여 얼마나 구조 정보를 변화시키지 않았는지를 계산. SSIM값이 높을 수록 원본 영상의 품질에 가깝다.

일단, 그냥 두 값이 높으면 좋은거다(원래 이미지와 비슷하다)고 알고 넘어가면 됨.

자세하게 알고싶다면...

[https://bskyvision.com/392](https://bskyvision.com/392)

[https://bskyvision.com/396](https://bskyvision.com/396)

```python
from skimage import data
import matplotlib.pyplot as plt

hr_cat = data.chelsea() # skimage에서 제공하는 예제 이미지를 불러옵니다.
hr_shape = hr_cat.shape[:2]

print(hr_cat.shape) # 이미지의 크기를 출력합니다.

plt.figure(figsize=(8,5))
plt.imshow(hr_cat)
```

```python
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

print("**동일 이미지 비교**")
print("PSNR :", peak_signal_noise_ratio(hr_cat, hr_cat))
print("SSIM :", structural_similarity(hr_cat, hr_cat, multichannel=True))
```

PSNR은 상한값이 없고, SSIM은 0~1 사이의 값을 가지기 때문에 각각 inf와 1이 계산됩니다.

(동일 이미지니까 최대 값이 나오는게 맞지 ㅇㅇ)

interpolation 방법에 대해 평가 척도를 대입해보자.

```python
import cv2

# 이미지를 특정 크기로 줄이고 다시 늘리는 과정을 함수로 정의합니다.
def interpolation_xn(image, n):
    downsample = cv2.resize(
        image,
        dsize=(hr_shape[1]//n, hr_shape[0]//n)
    )
    upsample = cv2.resize(
        downsample,
        dsize=(hr_shape[1], hr_shape[0]),
        interpolation=cv2.INTER_CUBIC
    )
    return upsample

lr2_cat = interpolation_xn(hr_cat, 2) # 1/2로 줄이고 다시 복원
lr4_cat = interpolation_xn(hr_cat, 4) # 1/4로 줄이고 다시 복원
lr8_cat = interpolation_xn(hr_cat, 8) # 1/8로 줄이고 다시 복원

images = [hr_cat, lr2_cat, lr4_cat, lr8_cat]
titles = ["HR", "x2", "x4", "x8"]

# 각 이미지에 대해 PSNR을 계산하고 반올림합니다.
psnr = [round(peak_signal_noise_ratio(hr_cat, i), 3) for i in images]
# 각 이미지에 대해 SSIM을 계산하고 반올림합니다.
ssim = [round(structural_similarity(hr_cat, i, multichannel=True), 3) for i in images]

# 이미지 제목에 PSNR과 SSIM을 포함하여 시각화 합니다. 
plt.figure(figsize=(16,10))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i])
    plt.title(titles[i] + f" [{psnr[i]}/{ssim[i]}]", fontsize=20)
```

![README/Untitled%204.png](README/Untitled%204.png)

⇒ 해상도를 줄일 수록 원래 크기로 interpoloation 했을 때, 계산 결과가 감소함을 알 수 있다.

(시각적으로도 그러하다)

## SRCNN 및 SRGAN 결과 비교하기

---

DIV2K 데이터셋 내에서 학습에 사용하지 않은 검증용 데이터셋을 이용하며, 몇 개 이미지만 뽑아서 Super Resolution을 진행한 후 특정 부분을 잘라내어 확대해봅시다.

```python
for i, (lr, hr) in enumerate(valid):
    if i == 12: break # 12번째 이미지를 불러옵니다.

lr_img, hr_img = np.array(lr), np.array(hr)

# bicubic interpolation
bicubic_img = cv2.resize(
    lr_img, 
    (hr.shape[1], hr.shape[0]), 
    interpolation=cv2.INTER_CUBIC
)

# 전체 이미지를 시각화합니다.
plt.figure(figsize=(20,15))
plt.subplot(311); plt.imshow(hr_img)

# SRCNN을 이용해 고해상도로 변환합니다.
srcnn_img = apply_srcnn(bicubic_img)

# SRGAN을 이용해 고해상도로 변환합니다.
srgan_img = apply_srgan(lr_img)

images = [bicubic_img, srcnn_img, srgan_img, hr_img]
titles = ["Bicubic", "SRCNN", "SRGAN", "HR"]

left_top = (700, 1090) # 잘라낼 부분의 왼쪽 상단 좌표를 지정합니다.

# bicubic, SRCNN, SRGAN 을 적용한 이미지와 원래의 고해상도 이미지를 시각화합니다.
plt.figure(figsize=(20,20))
for i, pind in enumerate([321, 322, 323, 324]):
    plt.subplot(pind)
    plt.imshow(crop(images[i], left_top, 200, 350))
    plt.title(titles[i], fontsize=30)
```

![README/Untitled%205.png](README/Untitled%205.png)

⇒ SRGAN의 결과가 매우 비슷하다!

각 SR 결과와 HR 이미지 사이의 PSNR, SSIM 계산해보기

```python
for i, (lr, hr) in enumerate(valid):
    if i == 24: break
    
lr_img, hr_img = np.array(lr), np.array(hr)
bicubic_img = cv2.resize(
    lr_img,
    (hr.shape[1], hr.shape[0]),
    interpolation=cv2.INTER_CUBIC
)

srcnn_img = apply_srcnn(bicubic_img)
srgan_img = apply_srgan(lr_img)

images = [bicubic_img, srcnn_img, srgan_img, hr_img]
titles = ["Bicubic", "SRCNN", "SRGAN", "HR"]

# 각 이미지에 대해 PSNR을 계산하고 반올림합니다.
psnr = [round(peak_signal_noise_ratio(hr_img, i), 3) for i in images]
# 각 이미지에 대해 SSIM을 계산하고 반올림합니다.
ssim = [round(structural_similarity(hr_img, i, multichannel=True), 3) for i in images]

# 이미지 제목에 PSNR과 SSIM을 포함하여 시각화 합니다. 
plt.figure(figsize=(18,13))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i])
    plt.title(titles[i] + f" [{psnr[i]}/{ssim[i]}]", fontsize=30)
```

![README/Untitled%206.png](README/Untitled%206.png)

SRGAN 의 정량적 결과가 더 낮게 나온다?

SRCNN의 학습에 Mean Squared Error를 사용했기 때문에, 생성해야 할 픽셀 값들을 고해상도 이미지와 비교해 단순히 **평균적으로 잘 맞추는 방향**으로 예측했기 때문입니다. 이러한 문제는 SRCNN 뿐만 아니라 **MSE만을 사용해 학습하는 대부분의 신경망에서 발생하는 현상**이기도 합니다.

![README/Untitled%207.png](README/Untitled%207.png)

SRGAN 결과의 경우 **매우 선명하게 이상함**을 확인할 수 있습니다. 이는 Generator가 고해상도 이미지를 생성하는 과정에서 Discriminator를 속이기 위해 이미지를 **진짜 같이 선명하게 만들도록 학습 되었기 때문**입니다. 추가로 앞서 설명했듯이 VGG구조를 이용한 content loss를 통해 학습한 것 또한 사실적인 이미지를 형성하는데 크게 기여했다고 합니다. 다만, 입력되었던 저해상도 이미지가 매우 제한된 정보를 가지고 있기에 고해상도 이미지와 세부적으로 동일한 모양으로 선명하진 않은 것

⇒ 확실히 선명하긴한데, 창문 모양들이 다 이상함. 좀 기괴한 모양...

![README/Untitled%208.png](README/Untitled%208.png)

SRGAN이 더 선명하기는 하지만, 자세히 모양을 살펴보면, 지붕이 무슨 파인애플모양처럼 변형되어있음.

## 학습한 내용 정리하기

---

이전에 학습했던 Super Resolution 방법들은 세부적으로 **SISR(Single Image Super Resolution)** 방법에 속합니다. (Interpolation, SRCNN, SRGAN)

**RefSR(Reference-based Super Resolution) : '**여러 장의 저해상도 이미지를 잘 활용하여 고해상도 이미지를 생성한다면 더 좋은 결과를 얻을 수 있지 않을까??' 하는 질문에서 생겨난 방법.

모이 저해상도 이미지만을 받아서 고해상도 이미지를 생성하는 것이 아니라, 해상도를 높이는 데 참고할 만한 다른 이미지를 같이 제공해 주는 것이죠.

**'Image Super-Resolution by Neural Texture Transfer' 논문에서 내놓은 결과**

![README/Untitled%209.png](README/Untitled%209.png)

`Ref images`라고 나타난 두 이미지를 입력으로 Super Resolution을 수행했으며, `SRNTT`라고 나타난 결과를 보면 이전에 학습했던 SRGAN 보다 훨씬 더 선명한 것을 확인할 수 있습니다!

### 차별을 재생성하는 인공지능

인공지능이 가진 문제들 중 하나는 차별에 관한 것

학습에 사용하는 데이터는 인공지능 학습에 가장 중요한 영향을 주는 몇 가지 요인들 중 하나.

이러한 데이터가 편향(bias)을 담고 있다면 그로 부터 학습된 인공지능은 아무렇지도 않게 차별을 하게 됨.

Super Resolution 문제에서도 이러한 차별 사례가 있기 때문

문제가 된 방법은 2020년 초에 발표된 PULSE 라는 구조 입니다. 우선 이 방법은 아래 그림과 같이 Super Resolution 문제에서 매우 좋은 성능을 갖습니다. 저해상도 이미지 내에 거의 없는 정보를 완전히 새롭게 생성하는 수준. *문제는 저해상도의 얼굴 이미지를 입력했을 때, 모두 백인 으로 만들어 낸다는 것.*

[공정한 AI 얼굴인식기]

[https://www.kakaobrain.com/blog/57](https://www.kakaobrain.com/blog/57)

[Single Image Super Resolution using Deep Learning Overview]

[https://hoya012.github.io/blog/SIngle-Image-Super-Resolution-Overview/](https://hoya012.github.io/blog/SIngle-Image-Super-Resolution-Overview/)

ill-posed problem

![README/Untitled%2010.png](README/Untitled%2010.png)

저해상도 이미지를 만들 때 사용한 distortion, down sapling 기법이 무엇이었는지에 따라 Super Resolution의 성능이 달라질 수 있습니다.

일반적으로 논문 들에서는 bicubic down sampling한 데이터셋을 사용하고 있으며, 실생활에서 Super Resolution을 사용하고자 하는 경우에는 unknown down sampling도 고려를 해야 함을 알 수 있습니다.

일반적으로 SIngle Image Super Resolution 문제를 접근하는 방식은 크게 3가지가 존재합니다.

- Interpolation-based method
    - 말 그대로 이미지를 크게 만들어줄 뿐, 디테일한 부분은 blur가 존재하거나 열화가 존재하는 것을 확인할 수 있습니다.
- Reconstruction-based method
- (Deep) Learning-based method

### 딥러닝을 이용한 Single Image Super Resolution

***SRCNN***

단 3개의 convolutional layer만 사용하였으며 딥러닝을 적용하지 않은 방법들에 비해 높은 성능 수치를 보이며 Super Resolution 분야에도 딥러닝을 적용할 수 있다는 가능성을 보인 논문.

흥미로운 점은 architecture를 구성할 때, 각각 convolutional layer가 가지는 의미를 전통적인 Super Resolution 관점에서 해석하고 있으며 각각 layer가 patch extraction, nonlinear mapping, reconstruction을 담당하고 있다고 서술

**FSRCNN**

![README/Untitled%2011.png](README/Untitled%2011.png)

Input으로 들어가는 LR 이미지를 그대로 convolution layer에 집어넣는 방식을 사용하였고, 마지막에 feature map의 가로, 세로 크기를 키워주는 deconvolution 연산을 사용하여 HR 이미지로 만드는 것이 가장 큰 특징.

LR 이미지를 convolution 연산을 하게 되면 키워주고자 하는 배수에 제곱에 비례하여 연산량이 줄어들게 됩니다.

거의 실시간에 준하는 성능을 보일 수 있음을 강조.

연산량이 줄어든 만큼 convolution layer의 개수도 늘려주면서 정확도(PSNR)도 챙길 수 있음.

### ESPCN

“Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network, 2016 CVPR”

**sub-pixel convolutional layer** 라는 구조를 제안하였으며, 이 연산은 **pixel shuffle** 혹은 **depth to space** 라는 이름으로도 불리는 연산.

![README/Untitled%2012.png](README/Untitled%2012.png)

만약 r배의 up scaling을 하고자 하는 경우, Convolutional layer를 거친 뒤 마지막 layer에서 feature map의 개수를 r 제곱개로 만들어 준 뒤, 각 feature map의 값들을 위의 그림처럼 순서대로 배치하여 1 채널의 HR 이미지로 만들어 주게 됩니다.

### VDSR

deep한 구조를 제안한 논문이 2016년 공개.

[https://cv.snu.ac.kr/research/VDSR/](https://cv.snu.ac.kr/research/VDSR/)

20-layer convolutional network를 제안하였고, 원활한 학습을 위해 input image를 최종 output에 더해주는 방식인 residual learning 을 사용

초기에 높은 learning rate를 사용하여 수렴이 잘 되도록 하기 위해 gradient clipping 도 같이 수행

VDSR 이후 이루어진 시도들

![README/Untitled%2013.png](README/Untitled%2013.png)

### SRGAN

기존 Super Resolution 들은 MSE loss를 사용하여 복원을 하다보니 PSNR 수치는 높지만 다소 blurry 한 output을 내고 있음을 지적하며, 사람 눈에 그럴싸하게 보이는 복원을 하기 위해 GAN을 접목시키는 방법을 제안.

![README/Untitled%2014.png](README/Untitled%2014.png)

실험 결과에서는 PSNR 수치는 떨어지지만 사람 눈에 보기엔 보다 더 그럴싸한 결과를 낼 수 있음을 제시하고 있으며, 키워주고자 하는 배수가 커질수록 효과를 더 볼 수 있습니다.

## 프로젝트: SRGAN 활용하기

---

결과물: [https://github.com/bluecandle/2020_AIFFEL/blob/master/daily_notes/exploration_codes/e19_code/E19.ipynb](https://github.com/bluecandle/2020_AIFFEL/blob/master/daily_notes/exploration_codes/e19_code/E19.ipynb)

# GD Proj2

이미지 어디까지 우려볼까?

![https://i.redd.it/ypjkwzv6b0k51.png](https://i.redd.it/ypjkwzv6b0k51.png)

## **실습목표**

---

1. Augmentation을 모델 학습에 적용하기
2. Augmentation의 적용을 통한 학습 효과 확인하기
3. 최신 data augmentation 기법 구현 및 활용하기

## Augmentation 적용 (1) 데이터 불러오기

```python
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds

import urllib3
urllib3.disable_warnings()
(ds_train, ds_test), ds_info = tfds.load(
    'stanford_dogs',
    split=['train', 'test'],
    shuffle_files=True,
    with_info=True,
)

fig = tfds.show_examples(ds_info, ds_train)
```

## Augmentation 적용 (2) Augmentation 적용하기

기본 전처리 함수

```python
def normalize_and_resize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    image = tf.image.resize(image, [224, 224])
    return tf.cast(image, tf.float32) / 255., label
```

⇒ 이렇게 되면 이미지 변환의 결과로 리턴받은 이미지를 그 다음 전처리 함수의 입력으로 연거푸 재사용할 수 있는 구조가 되어 편리

`random_flip_left_right()`과 `random_brightness()`를 활용해 보겠습니다.

```python
def augment(image,label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    return image,label
```

apply_normalize_on_dataset()를 통해서 일반적인 전처리 과정, 즉 normalize, resize, augmentation과 shuffle을 적용하도록 하겠습니다.

주의해야할 점은 shuffle이나 augmentation은 테스트 데이터셋에는 적용하지 않아야 한다는 점

```python
# 데이터셋(ds)을 가공하는 메인함수
def apply_normalize_on_dataset(ds, is_test=False, batch_size=16, with_aug=False):
    ds = ds.map(
        normalize_and_resize_img,  # 기본적인 전처리 함수 적용
        num_parallel_calls=2
    )
    if not is_test and with_aug:
        ds = ds.map(
            augment,       # augment 함수 적용
            num_parallel_calls=2
        )
    ds = ds.batch(batch_size)
    if not is_test:
        ds = ds.repeat()
        ds = ds.shuffle(200)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds
```

***test-time augmentation (TTA)***

여러 결과를 조합하기 위한 앙상블(ensemble) 방법 중 하나로 테스트 데이터셋에 augmentation을 적용하는 test-time augmentation이라는 방법이 있습니다.

[https://hwiyong.tistory.com/215](https://hwiyong.tistory.com/215)

TTA는 증강된 이미지를 여러번 보여준 다음 각각의 단계에 대해서 prediction을 평균하고 이 결과를 최종값으로 사용하는 것

장점 : 1번만 예측하여 결과를 도출했을 시에 해당 차량의 클래스가 아니라고 추측하는 반면, TTA를 통해 도출된 최종 결과값은 올바른 차량의 클래스라고 맞춘다는 것

## Augmentation 적용 (3) 비교실험 하기

---

augmentation을 적용한 데이터를 학습시킨 모델과 적용하지 않은 데이터를 학습시킨 모델의 성능 비교.

텐서플로우 케라스의 ResNet50 중 imagenet에 훈련된 모델을 불러옵니다. `include_top`은 마지막 fully connected layer를 포함할지 여부.

해당 레이어를 포함하지 않고 생성하면 특성 추출기(feature extractor) 부분만 불러와 우리의 필요에 맞게 수정된 fully connected layer를 붙여서 활용할 수 있습니다.

이미지넷(ImageNet)과 우리의 테스트셋이 서로 다른 클래스를 가지므로, 마지막에 추가해야 하는 fully connected layer의 구조(뉴런의 개수) 또한 다르기 때문 (stanford_dog 데이터셋 사용중)

augmentation 적용 안한거.

```python
num_classes = ds_info.features["label"].num_classes
resnet50 = keras.models.Sequential([
    keras.applications.resnet.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224,3),
        pooling='avg',
    ),
    keras.layers.Dense(num_classes, activation = 'softmax')
])
```

augmentation 적용한 데이터셋으로 학습시킬 resnet 하나 더.

```python
aug_resnet50 = keras.models.Sequential([
    keras.applications.resnet.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224,3),
        pooling='avg',
    ),
    keras.layers.Dense(num_classes, activation = 'softmax')
])
```

위에서 정의한 `apply_normalize_on_dataset` 함수에 `with_aug` 파라미터를 각각 다르게 부여.

```python
(ds_train, ds_test), ds_info = tfds.load(
    'stanford_dogs',
    split=['train', 'test'],
    as_supervised=True,
    shuffle_files=True,
    with_info=True,
)
ds_train_no_aug = apply_normalize_on_dataset(ds_train, with_aug=False)
ds_train_aug = apply_normalize_on_dataset(ds_train, with_aug=True)
ds_test = apply_normalize_on_dataset(ds_test, is_test = True)
```

두 모델에 각각 augmentation이 적용된 데이터셋과 적용되지 않은 데이터셋 학습 후 검증 시작.

```python
tf.random.set_seed(2020)
resnet50.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.SGD(lr=0.01),
    metrics=['accuracy'],
)

aug_resnet50.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.SGD(lr=0.01),
    metrics=['accuracy'],
)

history_resnet50_no_aug = resnet50.fit(
    ds_train_no_aug, # augmentation 적용하지 않은 데이터셋 사용
    steps_per_epoch=int(ds_info.splits['train'].num_examples/16),
    validation_steps=int(ds_info.splits['test'].num_examples/16),
    epochs=20,
    validation_data=ds_test,
    verbose=1,
    use_multiprocessing=True,
)

history_resnet50_aug = aug_resnet50.fit(
    ds_train_aug, # augmentation 적용한 데이터셋 사용
    steps_per_epoch=int(ds_info.splits['train'].num_examples/16),
    validation_steps=int(ds_info.splits['test'].num_examples/16),
    epochs=20,
    validation_data=ds_test,
    verbose=1,
    use_multiprocessing=True,
)
```

훈련 과정 시각화

```python
plt.plot(history_resnet50_no_aug.history['val_accuracy'], 'r')
plt.plot(history_resnet50_aug.history['val_accuracy'], 'b')
plt.title('Model validation accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['No Augmentation', 'With Augmentation'], loc='upper left')
plt.show()
```

![README/Untitled%2015.png](README/Untitled%2015.png)

훈련 과정 조금 더 확대해서 살펴보자.

```python
plt.plot(history_resnet50_no_aug.history['val_accuracy'], 'r')
plt.plot(history_resnet50_aug.history['val_accuracy'], 'b')
plt.title('Model validation accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['No Augmentation', 'With Augmentation'], loc='upper left')
plt.grid(True)
plt.ylim(0.72, 0.76)
plt.show()
```

![README/Untitled%2016.png](README/Untitled%2016.png)

## 심화 기법 (1) Cutmix Augmentation

---

논문

[https://arxiv.org/pdf/1905.04899.pdf](https://arxiv.org/pdf/1905.04899.pdf)

캐글 노트북

[https://www.kaggle.com/cdeotte/cutmix-and-mixup-on-gpu-tpu](https://www.kaggle.com/cdeotte/cutmix-and-mixup-on-gpu-tpu)

This notebook shows how perform cutmix and mixup data augmentation using the GPU/TPU with TensorFlow.data.Dataset.

다른 코드 예시는 위 노트북 살펴보자.

CutMix는 네이버 클로바(CLOVA)에서 발표한 CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features 에서 제안된 방법

이름에서 알 수 있는 것처럼, 이미지 데이터를 자르고 섞는다고 생각할 수 있습니다.

Mixup은 특정 비율로 픽셀별 값을 섞는 방식이고, Cutout은 이미지를 잘라내는 방식입니다. CutMix는 Mixup과 비슷하지만 일정 영역을 잘라서 붙여주는 방법.

CutMix는 이미지를 **섞는 부분과 섞은 이미지에 맞추어 라벨을 섞는 부분을 포함**

![README/Untitled%2017.png](README/Untitled%2017.png)

### 1) 이미지 섞기

배치 내의 이미지를 두 개 골라서 섞어줍니다. 이때 이미지에서 잘라서 섞어주는 영역을 바운딩 박스(bounding box)라고 부릅니다.

tfds에서 한 장뽑고, 이미지 2개 얻기.

```python
import matplotlib.pyplot as plt

# 데이터셋에서 이미지 2개를 가져옵니다. 
for i, (image, label) in enumerate(ds_train_no_aug.take(1)):
    if i == 0:
        image_a = image[0]
        image_b = image[1]
        label_a = label[0]
        label_b = label[1]
        break

plt.subplot(1,2,1)
plt.imshow(image_a)

plt.subplot(1,2,2)
plt.imshow(image_b)
```

이미지 a를 바탕 이미지로 하고 거기에 삽입할 두번째 이미지b가 있을 때, a에 삽입될 영역의 **바운딩 박스의 위치**를 결정하는 함수.

```python
def get_clip_box(image_a, image_b, img_size=224):
    # get center of box
    x = tf.cast( tf.random.uniform([],0, img_size),tf.int32)
    y = tf.cast( tf.random.uniform([],0, img_size),tf.int32)

    # get width of box
    _prob = tf.random.uniform([],0,1)
    width = tf.cast(img_size * tf.math.sqrt(1-_prob),tf.int32)
    
    # clip box in image and get minmax bbox
    xa = tf.math.maximum(0, x-width//2)
    ya = tf.math.maximum(0, y-width//2)
    yb = tf.math.minimum(img_size, y+width//2)
    xb = tf.math.minimum(img_size, x+width//2)
    
    return xa, ya, xb, yb

xa, ya, xb, yb = get_clip_box(image_a, image_b)
print(xa, ya, xb, yb)
```

바탕이미지 a에서 바운딩 박스 바깥쪽 영역을, 다른 이미지 b에서 바운딩 박스 안쪽 영역을 가져와서 합치기.

```python
# mix two images
def mix_2_images(image_a, image_b, xa, ya, xb, yb, img_size=224):
    one = image_a[ya:yb,0:xa,:]
    two = image_b[ya:yb,xa:xb,:]
    three = image_a[ya:yb,xb:img_size,:]
    middle = tf.concat([one,two,three],axis=1)
    top = image_a[0:ya,:,:]
    bottom = image_a[yb:img_size,:,:]
    mixed_img = tf.concat([top, middle, bottom],axis=0)
    
    return mixed_img

mixed_img = mix_2_images(image_a, image_b, xa, ya, xb, yb)
plt.imshow(mixed_img.numpy())
```

### 2) 라벨 섞기

이미지 섞었으니 라벨도 섞어야지!

⇒ 내일 이어서