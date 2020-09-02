# AIFFEL_29일차 2020.09.02

Tags: AIFFEL_DAILY

### 일정

---

1. LSTM 발표 준비 보강
2. cs231n lecture 10 : RNN 풀잎 (발표 : LSTM)
3. LMS F-26
4. LMS F-27
5. 코드마스터 session6 문제풀이

# LSTM 발표 준비 보강

---

[CS231n 2017 lecture10](https://www.notion.so/CS231n-2017-lecture10-2a2dd09971224c63bceb6b5d5847ee16)

# [F-26] TF2 API 개요

---

Tensorflow도 V2가 되면서 pyTorch가 가진 장점을 대부분 흡수하고, Keras라는 pyTorch와 아주 닮은 API를 Tensorflow의 표준 API로 삼으면서, Google이 가진 분산환경을 다루는 엄청난 기술력과 결합하여 더욱 강력하게 진화한 딥러닝 프레임워크로 진화해 왔습니다.

## ***학습목표***

---

- *Tensorflow V2의 개요와 특징을 파악한다.*
- *Tensorflow V2의 3가지 주요 API 구성방식을 이해하고 활용할 수 있다.*
- *GradientTape를 활용해 보고 좀더 로우레벨의 딥러닝 구현방식을 이해한다.*

가장 어려운 단계는 아무래도 딥러닝 모델의 그래디언트를 수식으로 구하고 그것을 바탕으로 backward propagation을 구현하는 것일 것.

모델이 훨씬 복잡해진다면 그 복잡한 수식의 gradient를 구하기 위해 엄청나게 복잡한 미분식을 다루어야 할텐데 생각만 해도 아찔한 일.

### V1 와 V2

---

Tensorflow는 forward propagation 방향의 모델만 설계하면 그 모델의 gradient를 사전에 미리 구해둘 수 있습니다.

이것을 가능하게 하기 위해 Tensorflow는 초기 V1때부터 독특한 설계 사상을 보유했는데, 그것은 바로 ***Tensorflow를 거대한 노드-엣지 사이의 유향 비순환 그래프(Directed Acyclic Graph, DAG)***로 정의.

노드와 노드를 연결하는 매 엣지마다 chain-rule을 기반으로 gradient가 역방향으로 전파될 수 있다는 간단하면서도 강력한 아이디어 ⇒ 이런 방식을 Tensorflow의 Graph Mode 라고 함.

***근데 문제가 있음!***

딥러닝 모델을 구성하는 그래프를 그려나가는 부분과, 그 그래프 상에서 연산이 실제 진행되어과는 과정을 엄격하게 분리. ⇒ 여기에서 가장 중요한 것이 바로 'session'이라는 개념이었음.

⇒ 그래서, 그래프 사이에 벌어지는 모든 연산이 반드시 session.run() 안에서 수행되어야 했음.

⇒ 대규모 분산환경에서의 확장성과 생산성이라는 장점도 있었지만, Tensorflow V1은 기본적으로 사용하기 어려웠음. 지저분...

⇒ 결정적으로, 그래프를 다 만들어놓고 돌려봐야 비로소 모델 구성시의 문제가 드러나서, 문제가 발생했을 때 해결하기가 너무 어렵고 복잡했음.

***문제의 해결? V2, Eager Mode 의 수용***

pytorch 가 제안한 Eaget Mode 를 통해 딥러닝 그래프가 다 그려지지 않아도 부분 실행 및 오류검증 가능.

⇒ TF도 수용.

그리고 Keras 라는 ML 프레임워크 수용.

V1 대비 V2의 장점 요약
쉬운 사용성, 쉬운 실행, 모델 설계 및 배포의 용이함, 데이터 파이프라인 간소화

# Tensorflow2 API로 모델 구성, 작성

---

딥러닝 모델을 다양한 방법으로 작성 가능하다.

⇒ 경우에 따라 적합한 모델링 방식을 택해서 사용할 수 있다는 매우 강력한 장점!

***세 가지 방법이 존재한다 : Sequential , Functional, Model Subclassing***

Functional은 Sequential의 보다 일반화된 개념입니다. 그리고 Subclassing은 클래스로 구현된 기존의 모델을 상속받아 자신만의 모델을 만들어나가는 방식.

### Sequential Model

---

Sequential 모델을 활용하면 손쉽게 딥러닝 모델을 쌓아나갈 수 있습니다. 입력부터 출력까지 레이어를 그야말로 sequential하게 차곡차곡 add해서 쌓아나가기만 하면 됩니다. 쉽다!

*하지만, 모델의 입력과 출력이 여러 개인 경우에는 적합하지 않은 모델링 방식*

⇒ Sequential 모델은 반드시 입력 1가지, 출력 1가지를 전제로 하기 때문.

### Functional API

---

```python
import tensorflow as tf
from tensorflow import keras

inputs = keras.Input(shape=(__원하는 입력값 모양__))
x = keras.layers.__넣고싶은 레이어__(관련 파라미터)(input)
x = keras.layers.__넣고싶은 레이어__(관련 파라미터)(x)
outputs = keras.layers.__넣고싶은 레이어__(관련 파라미터)(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.fit(x,y, epochs=10, batch_size=32)
```

Sequential Model을 활용하는 것과 다른 점은 바로 keras.Model을 사용한다는 점

그래서 Funtional API가 Sequential Model을 쓰는 것보다 더 일반적인 접근인 것입니다.

Sequential Model이란 사실 keras.Model을 상속받아 확장한 특수사례에 불과한 것이니까!

Functional API를 활용하면 앞서 배운 Sequential Model을 활용하는 것보다 더 자유로운 모델링 가능.

Functional 이라는 말 자체가 함수형으로 모델을 구성하겠다는거잖아. 그래서, **입력과 출력을 규정하여 모델 전체를 규정**한다는 것임!

⇒ 다중 입,출력을 가지는 모델을 구성할 수 있음.

### Subclassing

---

```python
import tensorflow as tf
from tensorflow import keras

class CustomModel(keras.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.__정의하고자 하는 레이어__()
        self.__정의하고자 하는 레이어__()
        self.__정의하고자 하는 레이어__()

    def call(self, x):
        x = self.__정의하고자 하는 레이어__(x)
        x = self.__정의하고자 하는 레이어__(x)
        x = self.__정의하고자 하는 레이어__(x)

        return x

model = CustomModel()
model.fit(x,y, epochs=10, batch_size=32)
```

가장 자유로운 모델링. 본질적으로는 Functional 접근과 차이가 없음.

(똑같이 keras.Model을 상속받은 모델 클래스를 만드는 것이기 때문.

1. **init**()이라는 메소드 안에서 레이어 구성을 정의합니다.
2. 그리고 **call()**이라는 메소드 안에서 레이어간 forward propagation을 구현

## V2 API로 모델 작성(1) MNIST , Sequential API 활용

---

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# 데이터 구성부분
mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train=x_train[...,np.newaxis]
x_test=x_test[...,np.newaxis]

print(len(x_train), len(x_test))

# Sequential Model을 구성해주세요.
"""
Spec:
1. 32개의 채널을 가지고, 커널의 크기가 3, activation function이 relu인 Conv2D 레이어
2. 64개의 채널을 가지고, 커널의 크기가 3, activation function이 relu인 Conv2D 레이어
3. Flatten 레이어
4. 128개의 아웃풋 노드를 가지고, activation function이 relu인 Fully-Connected Layer(Dense)
5. 데이터셋의 클래스 개수에 맞는 아웃풋 노드를 가지고, activation function이 softmax인 Fully-Connected Layer(Dense)
"""

# 여기에 모델을 구성해주세요
model = keras.Sequential()
model.add(keras.layers.Conv2D(32,3,activation='relu'))
model.add(keras.layers.Conv2D(64,3,activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128,activation='relu'))
model.add(keras.layers.Dense(10, activation = 'softmax'))

# 모델 학습 설정

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)
```

## V2 API로 모델 작성(2) MNIST , Functional API 활용

---

keras.Model을 직접 활용하여야 하므로,

keras.Input으로 정의된 input및 output 레이어 구성을 통해 model을 구현해야함.

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train=x_train[...,np.newaxis]
x_test=x_test[...,np.newaxis]

print(len(x_train), len(x_test))

from tensorflow.keras import layers
"""
Spec:
0. (28X28X1) 차원으로 정의된 Input
1. 32개의 채널을 가지고, 커널의 크기가 3, activation function이 relu인 Conv2D 레이어
2. 64개의 채널을 가지고, 커널의 크기가 3, activation function이 relu인 Conv2D 레이어
3. Flatten 레이어
4. 128개의 아웃풋 노드를 가지고, activation function이 relu인 Fully-Connected Layer(Dense)
5. 데이터셋의 클래스 개수에 맞는 아웃풋 노드를 가지고, activation function이 softmax인 Fully-Connected Layer(Dense)
"""

# 여기에 모델을 구성해 주세요.

inputs = keras.Input(shape=(28,28,1))
conv32 = layers.Conv2D(32,3,activation='relu')
conv64 = layers.Conv2D(64,3,activation='relu')
flatten = layers.Flatten()
dense128 = layers.Dense(128, activation="relu")
dense10 = layers.Dense(10, activation = 'relu')

x = conv32(inputs)
x = conv64(x)
x = flatten(x)
x = dense128(x)
outputs = dense10(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
```

## V2 API로 모델 작성(3) MNIST , Subclassing 활용

---

keras.Model을 상속받은 클래스를 만드는 것.

1. **init**() 메소드 안에서 레이어를 선언
2. call() 메소드 안에서 forward propagation을 구현하는 방식

⇒ Functional 방식과 비교하자면, call()의 입력이 Input이고, call()의 리턴값이 Output

```python
# Subclassing을 활용한 Model을 구성해주세요.
"""
Spec:
0. keras.Model 을 상속받았으며, __init__()와 call() 메소드를 가진 모델 클래스
1. 32개의 채널을 가지고, 커널의 크기가 3, activation function이 relu인 Conv2D 레이어
2. 64개의 채널을 가지고, 커널의 크기가 3, activation function이 relu인 Conv2D 레이어
3. Flatten 레이어
4. 128개의 아웃풋 노드를 가지고, activation function이 relu인 Fully-Connected Layer(Dense)
5. 데이터셋의 클래스 개수에 맞는 아웃풋 노드를 가지고, activation function이 softmax인 Fully-Connected Layer(Dense)
6. call의 입력값이 모델의 Input, call의 리턴값이 모델의 Output
"""

# 여기에 모델을 구성해주세요
class CustomModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = keras.layers.Conv2D(32, 3, activation='relu')
        self.conv2 = keras.layers.Conv2D(64, 3, activation='relu')
        self.flatten = keras.layers.Flatten()
        self.fc1 = keras.layers.Dense(128, activation='relu')
        self.fc2 = keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

model = CustomModel()
```

⇒ 앞서 했던 세 가지 (Sequential, Functional, Subclassing) 모두 결과는 같다!

### CIFAR-100 의 문제점??

---

특징 : 클래스가 100개인데, 20개의 subclass로 나뉜다

각 클래스당 이미지 수가 총 600개(500 훈련, 100 테스트) 로 좀 적다??

CIFAR-10은 클래스당 6000개잖아.

## V2 API로 모델 작성(4) CIFAR-100 , Sequential API 활용

---

```python
# Sequential Model을 구성해주세요.
"""
Spec:
1. 16개의 채널을 가지고, 커널의 크기가 3, activation function이 relu인 Conv2D 레이어
2. pool_size가 2인 MaxPool 레이어
3. 32개의 채널을 가지고, 커널의 크기가 3, activation function이 relu인 Conv2D 레이어
4. pool_size가 2인 MaxPool 레이어
5. 256개의 아웃풋 노드를 가지고, activation function이 relu인 Fully-Connected Layer(Dense)
6. 데이터셋의 클래스 개수에 맞는 아웃풋 노드를 가지고, activation function이 softmax인 Fully-Connected Layer(Dense)
"""

# 여기에 모델을 구성해주세요
from tensorflow.keras import layers

model = keras.Sequential()
model.add(layers.Conv2D(16,3,activation='relu'))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Conv2D(32,3,activation='relu'))
model.add(layers.MaxPool2D(2,2))
# 주의! Flatten 넣어줘
model.add(layers.Flatten())
model.add(layers.Dense(256,activation = 'relu'))
model.add(layers.Dense(100,activation='softmax'))

# 모델 학습 설정

# 학습 관련 부분을 작성해주세요
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)
```

⇒ 결과 : (2.5883734226226807, 0.3571000099182129)

loss, accuracy

## V2 API로 모델 작성(5) CIFAR-100 , Functional API 활용

---

```python
# Functional API를 활용한 Model을 구성해주세요.
"""
Spec:
0. (32X32X3) 차원으로 정의된 Input
1. 16개의 채널을 가지고, 커널의 크기가 3, activation function이 relu인 Conv2D 레이어
2. pool_size가 2인 MaxPool 레이어
3. 32개의 채널을 가지고, 커널의 크기가 3, activation function이 relu인 Conv2D 레이어
4. pool_size가 2인 MaxPool 레이어
5. 256개의 아웃풋 노드를 가지고, activation function이 relu인 Fully-Connected Layer(Dense)
6. 데이터셋의 클래스 개수에 맞는 아웃풋 노드를 가지고, activation function이 softmax인 Fully-Connected Layer(Dense)
"""

# 여기에 모델을 구성해주세요
inputs = keras.Input(shape=(32, 32, 3))

x = keras.layers.Conv2D(16, 3, activation='relu')(inputs)
x = keras.layers.MaxPool2D((2,2))(x)
x = keras.layers.Conv2D(32, 3, activation='relu')(x)
x = keras.layers.MaxPool2D((2,2))(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(256, activation='relu')(x)
predictions = keras.layers.Dense(100, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=predictions)

# 모델 학습 설정

# 학습 관련 부분을 작성해주세요
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)
```

⇒ 결과 :(2.656320333480835, 0.3449000120162964)

loss, accuracy

## V2 API로 모델 작성(6) CIFAR-100 , Subclassing 활용

---

```python
/# Subclassing을 활용한 Model을 구성해주세요.
"""
Spec:
0. keras.Model 을 상속받았으며, __init__()와 call() 메소드를 가진 모델 클래스
1. 16개의 채널을 가지고, 커널의 크기가 3, activation function이 relu인 Conv2D 레이어
2. pool_size가 2인 MaxPool 레이어
3. 32개의 채널을 가지고, 커널의 크기가 3, activation function이 relu인 Conv2D 레이어
4. pool_size가 2인 MaxPool 레이어
5. 256개의 아웃풋 노드를 가지고, activation function이 relu인 Fully-Connected Layer(Dense)
6. 데이터셋의 클래스 개수에 맞는 아웃풋 노드를 가지고, activation function이 softmax인 Fully-Connected Layer(Dense)
7. call의 입력값이 모델의 Input, call의 리턴값이 모델의 Output
"""

# 여기에 모델을 구성해주세요

class CustomModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = keras.layers.Conv2D(16, 3, activation='relu')
        self.maxpool1 = keras.layers.MaxPool2D((2,2))
        self.conv2 = keras.layers.Conv2D(32, 3, activation='relu')
        self.maxpool2 = keras.layers.MaxPool2D((2,2))
        self.flatten = keras.layers.Flatten()
        self.fc1 = keras.layers.Dense(256, activation='relu')
        self.fc2 = keras.layers.Dense(100, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

model = CustomModel()

# 모델 학습 설정

# 학습 관련 부분을 작성해주세요
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)
```

⇒ 결과 :(2.507786512374878, 0.3727000057697296)

셋 다 결과는 비슷비슷하다.

## GradientTape

---

위의 작업들에서 모델 학습 관련 부분은 동일하게 했엇음. (모델 구성은 다른 방법을 썼었어도)

Numpy만 가지고 딥러닝을 구현하는 것을 회상해 봅시다. model.fit()이라는 한줄로 수행 가능한 딥러닝 모델 훈련 과정은 실제로는 어떠했나요?

- **Forward Propagation 수행 및 중간 레이어값 저장**
- **Loss 값 계산**
- **중간 레이어값 및 Loss를 활용한 체인룰(chain rule) 방식의 역전파(Backward Propagation) 수행**
- **학습 파라미터 업데이트**

이상 4단계로 이루어진 train_step 을 여러번 반복했습니다.

Tensorflow에서 제공하는 tf.GradientTape는 위와 같이 순전파(forward pass) 로 진행된 모든 연산의 중간 레이어값을 tape에 기록하고 gradient를 계산 후 tape를 폐기하는 기능을 수행.

tf.GradientTape는 이후 그래디언트를 좀더 고급스럽게 활용하는 다양한 기법을 통해 자주 만나게 될 것.

```python
loss_func = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# tf.GradientTape()를 활용한 train_step
def train_step(features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features)
        loss = loss_func(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
```

```python
import time
def train_model(batch_size=32):
    start = time.time()
    for epoch in range(5):
        x_batch = []
        y_batch = []
        for step, (x, y) in enumerate(zip(x_train, y_train)):
            if step % batch_size == batch_size-1:
                x_batch.append(x)
                y_batch.append(y)
								# 여기서 loss를 구하는데 위에서 정의한 함수가 쓰인다.
                loss = train_step(np.array(x_batch), np.array(y_batch))
                x_batch = []
                y_batch = []
        print('Epoch %d: last batch loss = %.4f' % (epoch, float(loss)))
    print("It took {} seconds".format(time.time() - start))

train_model()
```

```python
# evaluation
prediction = model.predict(x_test, batch_size=x_test.shape[0], verbose=1)
temp = sum(np.squeeze(y_test) == np.argmax(prediction, axis=1))
temp/len(y_test)  # Accuracy
```

⇒ 결과 : 0.0463

# [F-27] 컴퓨터 파워 UP

---

### **학습 목표**

---

- 멀티태스킹, 병렬프로그래밍과 동시성에 대해 이해합니다.

## 멀티태스킹이란?

---

어떻게 한정된 컴퓨팅 자원을 활용하여 여러 가지 일을 효율적으로 진행할 수 있을까요?

컴퓨터에서는 이렇게 여러 가지 작업을 동시에 하는 것을 "멀티태스킹"이라고 합니다.

### 동시성(Concurrency)

---

배달과 그릇 찾기라는 2가지 작업을 한다고 할 때, 한 명의 직원을 고용해도 사람이 바삐 움직여 주기만 하면 배달, 그릇찾기, 배달, 그릇 찾기 등 여러 가지 일을 동시에 수행할 수 있습니다.

물론 진짜로 동시에 하는 건 아니지만, 오늘 하루의 일과를 보고 받을 때에는 2가지 일이 동시에 종료가 되었다고 말하겠지요.

이렇게 ***정***말로 동시에 하는 건 아니지만 여러 가지 일을 빠르게 번갈아가며 수행해 동시에 수행하는 것처럼 일하는 것을 동시성이라고 합니다.

***하나의 주체가,*** ***정말로 동시에 하는 건 아니지만 여러 가지 일을 빠르게 번갈아가며 수행해 동시에 수행하는 것처럼 일하는 것***

### 병렬성(Parallelism)

---

업무를 분담해서 할 수도 있을 겁니다.

두 명의 직원을 고용해서 한 명은 그릇 찾기 일만 하고 한 명은 배달 일만 하는 거예요.

이런 경우에는 정말 동시에 2가지 일이 진행될 수 도 있겠네요.

이 경우 일이 병렬적으로 처리된다고 말합니다.

여러 주체가 같이 일을 하여 동시에 일을 처리하는 것.

### 동기 vs 비동기 (Synchronous vs Asynchronous)

---

어떤 일을 바로 하지 못하고 대기해야 하는 일을 일컬을 때, 

컴퓨터에서는 **"바운드(bound)되었다"** 라는 표현을 많이 씁니다.

바운드되고 있으면 이걸 계속 기다려야 할지 아니면 종료되는 사이에 다른 걸 실행하는것이 좋을지 고민.

앞 작업이 종료되기를 무조건 기다렸다가 다음 작업을 수행하는 것은 동기(synchronized) 방식

기다리는 동안 다른 일을 처리하는 것을 비동기(asynchronous) 방식

![images/Untitled.png](images/Untitled.png)

### **I/O Bound vs CPU Bound**

---

컴퓨터가 일을 수행하면서 뭔가 기다릴 때, 즉 속도에 제한이 걸릴 때는 2가지 경우에 해당하는 경우가 대부분입니다.

- I/O 바운드: 입력과 출력에서의 데이터(파일)처리에 시간이 소요될 때.
- CPU 바운드: 복잡한 수식 계산이나 그래픽 작업과 같은 엄청난 계산이 필요할 때.

[https://stackoverflow.com/questions/868568/what-do-the-terms-cpu-bound-and-i-o-bound-mean](https://stackoverflow.com/questions/868568/what-do-the-terms-cpu-bound-and-i-o-bound-mean)

### 프로세스, 쓰레드, 프로파일링

---

### 프로세스

---

An Instance of program( ex. Chrome )
프로그램을 구동하여 프로그램 자체와 프로그램의 상태가 메모리 상에서 실행되는 작업 단위

프로그램을 실행하면, OS에서 프로세스를 생성함!

(하나의 프로그램을 여러 번 실행하면 여러 개의 프로세스가 생김)

그리고 프로세스는 OS의 커널에서 시스템 자원(CPU,mem) 및 자료구조를 이용함.

### Thread(쓰레드)

---

프로그램, 특히 프로세스 내에서 실행되는 흐름의 단위.

요리를 만드는 프로그램이라고 한다면 김밥, 떡볶이를 만드는 각각의 요리라는 프로세스에도, 밥짓기, 재료 볶기, 끓이기 등등의 작업을 스레드에 비유할 수 있습니다.

같은 작업을 좀 더 빠르게 처리하기 위해 여러 개의 스레드를 생성하기도 합니다.

![images/Untitled%201.png](images/Untitled%201.png)

프로세스는 김밥, 떡볶이를 만드는 각각의 요리사와 같습니다.

이들은 각자의 전용 주방공간(Heap)에서 밥짓기, 재료 볶기, 끓이기 등등의 작업을 병렬적으로 수행합니다. 도마, 불판 등 주방공간(각 요리사의 주방공간)은 각각의 작업에 공유되지만, 요리사끼리 주방공간을 공유하지는 않습니다.

마찬가지로 프로세스도 자신만의 전용 메모리공간(Heap)을 가집니다. 이때 해당 프로세스 내의 스레드들은 이 메모리공간을 공유합니다. 그러나 다른 프로세스와 공유하지는 않습니다.

### 프로파일링

---

코드에서 시스템의 어느 부분이 느린지 혹은 어디서 RAM을 많이 사용하고 있는지를 확인하고 싶을 때 사용하는 기법

```python
import timeit
        
def f1():
    s = set(range(100))

    
def f2():
    l = list(range(100))

    
def f3():
    t = tuple(range(100))

def f4():
    s = str(range(100))

    
def f5():
    s = set()
    for i in range(100):
        s.add(i)

def f6():
    l = []
    for i in range(100):
        l.append(i)
    
def f7():
    s_comp = {i for i in range(100)}

    
def f8():
    l_comp = [i for i in range(100)]
    

if __name__ == "__main__":
    t1 = timeit.Timer("f1()", "from __main__ import f1")
    t2 = timeit.Timer("f2()", "from __main__ import f2")
    t3 = timeit.Timer("f3()", "from __main__ import f3")
    t4 = timeit.Timer("f4()", "from __main__ import f4")
    t5 = timeit.Timer("f5()", "from __main__ import f5")
    t6 = timeit.Timer("f6()", "from __main__ import f6")
    t7 = timeit.Timer("f7()", "from __main__ import f7")
    t8 = timeit.Timer("f8()", "from __main__ import f8")
    print("set               :", t1.timeit(), '[ms]')
    print("list              :", t2.timeit(), '[ms]')
    print("tuple             :", t3.timeit(), '[ms]')
    print("string            :", t4.timeit(), '[ms]')
    print("set_add           :", t5.timeit(), '[ms]')
    print("list_append       :", t6.timeit(), '[ms]')
    print("set_comprehension :", t5.timeit(), '[ms]')
    print("list_comprehension:", t6.timeit(), '[ms]')
```

⇒ 간단하게 생각하자면, 이렇게 시간을 측정하여 함수의 성능을 측정해보는 것도 포함된다!

근데, 더 엄밀히 말하자면, ***어플리케이션에서 자원이 가장 집중되는 지점을 찾아내는 기법***

프로파일러: 어플리케이션을 실행시키고 각각의 함수 실행에 소요되는 시간을 찾아내는 프로그램.

즉, 코드의 병목을 찾아내고 성능을 측정해주는 도구이다!

⇒ profile, cProfile 모듈 및 line_profiler 패키지를 참조하면 더 좋다!

### Scale Up vs Scale Out

---

scale up : 하나의 컴퓨터 성능을 최적화

scale out : 여러대의 컴퓨터를 하나처럼 사용하기.

![images/Untitled%202.png](images/Untitled%202.png)

[https://hyuntaeknote.tistory.com/m/4](https://hyuntaeknote.tistory.com/m/4)

스케일업 장점

---

1. 별도의 서버를 추가하지 않기 때문에 여러 대의 서버를 관리하면서 발생하는 데이터 정합성 이슈에서 자유롭다.
2. 서버를 한 대로 관리하면 **별도의 소프트웨어 라이선스 추가 비용이 발생하지 않습니다**.
3. 하드웨어를 추가하고 교체하는 작업이기 때문에 **구현이 어렵지 않습니다.**

스케일업 단점

---

1. 설치 가능한 CPU, mem, 디스크 수의 제한
2. 일정 수준이 넘어가는 순간 성능 증가 폭 미미.
3. 성능 증가 대비 업그레이드 비용 많이 증가.
4. 서버 한 대가 모든 클라이언트 트래픽을 감당하다가 터져버리면, 서버 복구 때까지 서비스를 중단해야함.

⇒ 규모가 커진다면, 스케일 아웃을 도입해야한다!

스케일아웃 장점

---

1. 하나의 노드에서 장애가 발생하더라도 다른 노드에서 서비스 제공이 가능하여 가용성을 높일 수 있다.
2. 필요에 따라 더 많은 서버를 추가하거나 감소시킬 수 있음. 확장에 유연함.
3. 로드 밸런싱을 통해 단일 서버에 작업이 쌓여서 멈춰있는 병목현상을 줄일 수 있음.

스케일아웃 단점

---

1. 소프트웨어 라이선스 비용 증가
2. 데이터 불일치가 잠재적으로 발생할 수 있음

[ex. 세션 인증 오류 예시]

![images/Untitled%203.png](images/Untitled%203.png)

1. User1의 경우 로그인 시, 최초 접속한 서버(그림에서는 WAS_1)의 세션 저장소에 세션을 생성하고 로그인 정보를 저장합니다.
2. 이후에 세션 ID를 클라이언트에게 발급하게 되고, 이를 클라이언트의 쿠키 저장소에 저장하여 클라이언트에서는 다음 요청을 보낼 때마다 이를 꺼내서 HTTP 요청 헤더에 실어서 보내게 됩니다.
3. 단일 서버의 경우에는 문제없이 동작하지만 그림과 같이 서버가 2대 이상 늘어날 시에는 WAS_1에서 발급받은 세션 ID를 WAS_2에서 인증하게 되면 WAS_2 에는 존재하지 않는 세션이므로 인증이 처리되지 않습니다.

스케일 업이 좋은 경우

---

빈번한 갱신이 일어나는 가운데 철저하게 ACID를 지켜야만 하는 RDB같은 시스템에 적합. 스케일 아웃은 어쨌든 정합성 이슈에서 완전히 자유로울 수는 없기 때문.

스케일 아웃이 좋은 경우

---

각각의 트랜잭션 처리는 비교적 단순하지만 다수의 처리를 동시 병행적으로 실시하지 않으면 안 되는 경우에 적절. 제일 대표적인 예시가 웹 서버! 다수의 요구를 동시 병행하여 처리할 필요가 있지만, 각각 처리는 비교적 단순하기 때문.

## 파이썬에서 멀티스레드 사용하기

---

### 스레드 생성

---

파이썬에서 멀티스레드의 구현은 threading 모듈을 이용

[https://docs.python.org/3/library/threading.html](https://docs.python.org/3/library/threading.html)

```python
from threading import *

class Delivery(Thread):
	def run(self):
		print("delivery")

class RetriveDish(Thread):
	def run(self):
		print("Retriving Dish")

work1 = Delivery()
work2 = RetriveDish()

def main():
	work1.run()
	work2.run()

if __name__ == '__main__':
    main()
```

### 쓰레드 생성 및 사용

---

thread 모듈의 Thread 클래스를 상속받아서 구현할 수도 있지만 그대로 인스턴스화 하여 스레드를 생성할 수도 있습니다.

인스턴스화 하려면 Thread 클래스에 **인자로 target**과 **args 값**을 넣어 줍니다. **args에 넣어 준 파라미터는 스레드 함수의 인자로 넘어갑니다.**

```python
t = Thread(target=함수이름, args=())
```

Thread 클래스에는 start(), join() 같은 스레드 동작 관련 메소드가 있습니다.

```python
from threading import *
from time import sleep

Stopped = False

def worker(work, sleep_sec):    # 일꾼 스레드입니다.
    while not Stopped:   # 그만 하라고 할때까지
        print('do ', work)    # 시키는 일을 하고
        sleep(sleep_sec)    # 잠깐 쉽니다.
    print('retired..')           # 언젠가 이 굴레를 벗어나면, 은퇴할 때가 오겠지요?
        
t = Thread(target=worker, args=('Overwork', 3))    # 일꾼 스레드를 하나 생성합니다. 열심히 일하고 3초간 쉽니다.
t.start()    # 일꾼, 이제 일을 해야지? 😈
```

스레드 함수가 루프를 돌때는 꼭 멈춰야 할지를 체크하는 flag(여기서는 Stopped)를 체크하도록 설계해야 합니다.

```python
# 이 코드 블럭을 실행하기 전까지는 일꾼 스레드는 종료하지 않습니다. 
Stopped = True    # 일꾼 일 그만하라고 세팅해 줍시다. 
t.join()                    # 일꾼 스레드가 종료할때까지 기다립니다. 
print('worker is gone.')
```

## 파이썬에서 멀티프로세스 사용하기

---

파이썬에서 멀티프로세스의 구현은 multiprocessing 모듈을 이용해서 할 수 있습니다.

[https://docs.python.org/3/library/multiprocessing.html](https://docs.python.org/3/library/multiprocessing.html)

```python
import multiprocessing as mp

def delivery():
    print('delivering...')

p = mp.Process(target=delivery, args=())
p.start() # 프로세스 시작
# p.join() # 실제 종료까지 기다림 (필요시에만 사용)
# p.terminate() # 프로세스 종료
```

## 파이썬에서 스레드/프로세스 풀 사용하기

---

사실 멀티스레드/프로세스 작업을 할 때 가장 많은 연산이 필요한 작업은 바로 이런 스레드나 프로세스를 생성하고 종료하는 일.

특히 스레드/프로세스를 사용한 뒤에는 제대로 종료해 주어야 컴퓨팅 리소스가 낭비되지 않습니다.

하나씩 하나씩 실행한다고 전체적인 프로그램의 성능이 좋아지지는 않아요. 오히려 더 번거로울 수 있음.

***그래서 실제로 사용할 때에는 스레드/프로세스 풀을 사용해서 생성합니다.***

### Pool을 만드는 두 가지 방법 중 futures 라이브러리 사용해보기.

---

1. Queue를 이용하여 스스로 만들기

[https://docs.python.org/3/library/queue.html](https://docs.python.org/3/library/queue.html)

2. concurrent.futures 라이브러리를 이용하는 방법.

[https://docs.python.org/ko/3/library/concurrent.futures.html](https://docs.python.org/ko/3/library/concurrent.futures.html)

파이썬 3.2부터 추가된 모듈.

한국말로 "동시성 퓨처"라고 번역해서 부르기도 하는데, 기능은 크게 4가지가 있습니다.

- **`Executor`** 객체
- **`ThreadPoolExecutor`** 객체
- **`ProcessPoolExecutor`** 객체
- **`Future`** 객체

`ThreadPoolExecutor`

---

Executor 객체를 이용하면 스레드 생성, 시작, 조인 같은 작업을 할 때, *with 컨텍스트 관리자*와 같은 방법으로 가독성 높은 코드를 구현할 수 있습니다. (프로세스 구현 방법 역시 동일하므로 설명은 스레드로만 하겠습니다.)

```python
# 예시 형태
with ThreadPoolExecutor() as executor:
    future = executor.submit(함수이름, 인자
```

```python
# 예시
from concurrent.futures import ThreadPoolExecutor

class Delivery:
    def run(self):
        print("delivering")
w = Delivery()

with ThreadPoolExecutor() as executor:
    future = executor.submit(w.run)
```

`multiprocessing.Pool`

---

`multiprocessing.Pool.map` 을 통해 여러개의 프로세스에 특정 함수를 매핑해서 병렬처리하도록 구현하는 방법이 널리 사용됨.

```python
from multiprocessing import Pool
from os import getpid

def double(i):
    print("I'm process ", getpid())    # pool 안에서 이 메소드가 실행될 때 pid를 확인해 봅시다.
    return i * 2

with Pool() as pool:
      result = pool.map(double, [1, 2, 3, 4, 5])
      print(result)
```

⇒ 깔끔하게 병렬처리가 가능한 좋은 코드 예시! 활용도가 높을 것으로 생각된다.

## 실전 예제

---

[https://docs.python.org/ko/3/library/concurrent.futures.html](https://docs.python.org/ko/3/library/concurrent.futures.html)

concurrent.modules 에는 두 가지 객체가 있음

1. Executor 객체
    1. Executor 아래에 ThreadPoolExecutor
    2. Executor 아래에 ProcessPoolExecutor
    3. 제공하는 세 가지 메소드 : submit, map, shutdown
2. Future 객체

`ProcessPoolExecutor`

---

ProcessPoolExecutor를 이용해서 멀티프로세스를 구현을 연습해보자!

```python
import math
import concurrent

PRIMES = [
    112272535095293,
    112582705942171,
    112272535095293,
    115280095190773,
    115797848077099,
    1099726899285419]

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
		# 제곱근 씌우고 반올림 한 값
    sqrt_n = int(math.floor(math.sqrt(n)))
		# 3부터 위에서 구한 값까지(+1 해줬으니) 2씩 증가시키며 i를 꺼낸다.
    for i in range(3, sqrt_n + 1, 2):
				# 꺼낸 i로 나눈 나머지가 0이면 소수가 아님 (나머지가 있어야 소수)
        if n % i == 0:
            return False
		# 다 통과하면 소수다.
    return True
```

맵-리듀스(map-reduce)스타일로 코드를 작성하고 map() 함수를 `ProcessPoolExecutor()` 인스턴스에서 생성된 executor 에서 실행 시킵니다.

```python
def main():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for number, prime in zip(PRIMES, executor.map(is_prime, PRIMES)):
            print('%d is prime: %s' % (number, prime))

if __name__ == '__main__':
    main()
```

병렬처리와 단일처리의 비교를 위해 코드를 아래와 같이 수정해보자.

- 프로파일링을 위한 시간계산 코드를 추가
- 단일처리로 수행했을 때의 코드를 추가, 단일처리 프로파일링을 위한 시간계산 코드를 추가.

```python
import time

def main():
    print("병렬처리 시작")
    start = time.time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for number, prime in zip(PRIMES, executor.map(is_prime, PRIMES)):
            print('%d is prime: %s' % (number, prime))
    end = time.time()
    print("병렬처리 수행 시각", end-start, 's')
		
		print("======================")

		print("단일처리 시작")
    start = time.time()
    for number, prime in zip(PRIMES, map(is_prime, PRIMES)):
        print('%d is prime: %s' % (number, prime))
    end = time.time()
    print("단일처리 수행 시각", end-start, 's')
print(" ❣\n🌲🦕.......")
```