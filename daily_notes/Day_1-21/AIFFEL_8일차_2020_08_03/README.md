# AIFFEL_8일차 2020.08.03

Tags: AIFFEL_DAILY

### 일정

1. 코딩마스터 session 2
2. LMS F-6
3. LMS F-7

---

# 코딩마스터 session 2

### Stability

A sorting algorithm is said to be stable if it maintains the relative order of records in the case of equality of keys.

다음 주에는 해쉬 문제들을 풀어봅시다.

[https://programmers.co.kr/learn/courses/30/parts/12077](https://programmers.co.kr/learn/courses/30/parts/12077)

---

# [F-6] Data 어떻게 표현하면 좋을까? 배열(array)과 표(table)

### **학습 목표**

---

- 데이터를 배열로 저장하는 것에 대해 이해하고 list, numpy의 사용법을 학습합니다.
- 구조화된 데이터를 저장하는 것을 이해하고 dictonary와 pandas 사용법을 학습합니다.
- 이미지 데이터를 numpy 배열로 저장하는 것을 이해하고 그 사용법을 학습합니다.
- 학습한 자료 구조를 활용해서 통계 데이터를 어떻게 계산하는지 학습합니다.

자유도, 불편추정량

[https://hsm-edu.tistory.com/13](https://hsm-edu.tistory.com/13)

array, list

[https://blog.martinwork.co.kr/theory/2018/09/22/what-is-difference-between-list-and-array.html](https://blog.martinwork.co.kr/theory/2018/09/22/what-is-difference-between-list-and-array.html)

그래서 파이썬의 리스트는 좀 묘합니다. 자료구조상 linked list의 기능을 모두 가지고 있지만 실제로는 array로 구현되어 있다고 합니다. linked list와 다르게 파이썬 리스트는 element들이 연속된 메모리 공간에 배치되도록 구현되어 있기 때문입니다. 그렇게 보면 파이썬의 리스트는 list와 array의 장점을 모두 취한 형태라고 볼 수 있겠습니다.

## NumPy

---

**3) type**

NumPy 라이브러리 내부의 자료형들은 파이썬 내장함수와 동일합니다. 그러나 살짝 헷갈리는 기능이 있을 수 있어요. 바로 내장함수 **`type()`**과 **`dtype()`**메소드입니다.

파이썬의 최상위 클래스는 바로 object입니다. 그러므로 Numpy는 dtype을 object로 지정해서라도 행렬 내 dtype을 일치시킬 수 있게 됩니다.

```python
D = np.array([0,1,2,3,[4,5],6])
print(D.dtype)
print(D[0])
print(type(D[0]))
print(D[4])
print(type(D[4]))
# ===============
object
**0
<class 'int'>
[4, 5]
<class 'list'>**
```

D[0]에 해당하는 숫자 0은 여전히 행렬 안에 정수 0으로 들어가 있습니다. (dtype은 object 로 나오는데!)

이렇게 ndarray와 상수, 또는 서로 크기가 다른 ndarray끼리 산술연산이 가능한 기능을 브로드캐스팅이라고 합니다.

[https://numpy.org/devdocs/user/theory.broadcasting.html](https://numpy.org/devdocs/user/theory.broadcasting.html)

The stretching analogy is only conceptual. numpy is smart enough to use the original scalar value without actually making copies so that broadcasting operations are as memory and computationally efficient as possible.

**The Broadcasting Rule**
**In order to broadcast, the size of the trailing axes for both arrays in an operation must either be the same size or one of them must be one.**

아래 2가지는 기능면에서 동일합니다. 원소의 순서를 임의로 뒤바꾸어 줍니다.

```python
print(np.random.permutation(10))
print(np.random.permutation([0,1,2,3,4,5,6,7,8,9]))
```

```python
# 아래 기능들은 어떤 분포를 따르는 변수를 임의로 표본추출해 줍니다. 

# 이것은 정규분포를 따릅니다.
print(np.random.normal(loc=0, scale=1, size=5))    # 평균(loc), 표준편차(scale), 추출개수(size)를 조절해 보세요.

# 이것은 균등분포를 따릅니다. 
print(np.random.uniform(low=-1, high=1, size=5))  # 최소(low), 최대(high), 추출개수(size)를 조절해 보세요.
```

```python
# np.transpose는 행렬의 축을 어떻게 변환해 줄지 임의로 지정해 줄 수 있는 일반적인 행렬 전치 함수입니다. 
# np.transpose(A, (2,1,0)) 은 A.T와 정확히 같습니다.

B = np.transpose(A, (2,0,1))
print(A)             # A는 (2,3,4)의 shape를 가진 행렬입니다. 
print(B)             # B는 A의 3, 1, 2번째 축을 자신의 1, 2, 3번째 축으로 가진 행렬입니다.
print(B.shape)  # B는 (4,2,3)의 shape를 가진 행렬입니다.
```

A Visual Intro to NumPy and Data Representation

[http://jalammar.github.io/visual-numpy/](http://jalammar.github.io/visual-numpy/)

자연어의 numpy 를 이용한 표현

```
임베딩(Embedding)이라는 과정을 거쳐 ndarray로 표현될 수 있습니다.
블로그의 예시에서는 71,290개의 단어가 들어있는 (문장들로 이루어진) 데이터셋이 있을때, 이를 단어별로 나누고 0 - 71289로 넘버링 했습니다.
이를 토큰화 과정이라고 합니다. 그리고 이 토큰을 50차원의 word2vec embedding 을 통해  [batchsize, sequencelength, embedding_size]의 ndarray로 표현할 수 있습니다.
```

** 여기서 잠깐 **

```
img_arr = np.array(img)
```

가 정상동작했습니다. 의아합니다. img는 파이썬 리스트 타입이 아니라

```
PIL.JpegImagePlugin.JpegImageFile
```

라는 타입을 가지고 있습니다. 어떻게

```
np.array(img)
```

가 정상동작한 걸까요?

img는 PIL.Image.Image 라는 랩퍼 클래스를 상속받은 타입을 가지고 있습니다. PIL.Image.Image는 리스트를 상속받지 않았습니다. 하지만 이 클래스는 __array_interface__라는

[속성](https://numpy.org/doc/stable/reference/arrays.interface.html) 이 정의되어 있습니다. 이런 방식으로 Pillow 라이브러리는 손쉽게 이미지를 Numpy ndarray로 변환 가능하게 해줍니다.

### data augmentation

딥러닝에서 위와 같은 이미지 조작은 data_augmentation의 경우 많이 사용 됩니다. data_augmentation은 말 그대로 데이터를 증강하는 것으로 딥러닝에서 데이터 량을 늘릴때 사용되는 기법입니다. 추후 노드 상세 내용을 학습할 예정이오니 지금은 참고 링크만 방문해 보세요.

[https://www.tensorflow.org/tutorials/images/data_augmentation](https://www.tensorflow.org/tutorials/images/data_augmentation)

## Hash

---

 어떤 데이터의 값을 찾을 때 인덱스가 아닌 "한국", "미국" 등 키(key)를 사용해 데이터에 접근하는 데이터 구조를 hash라고 합니다.

> Hash란 Key와 Value로 구성되어 있는 자료 구조로 두개의 열만 갖지만 수많은 행을 가지는 구조체입니다.

![https://aiffelstaticprd.blob.core.windows.net/media/images/f-11-8.1.max-800x600.png](https://aiffelstaticprd.blob.core.windows.net/media/images/f-11-8.1.max-800x600.png)

해시는 다른 프로그래밍언어에서는 매핑(mapping), 연관배열(associative array) 등으로 불리고 파이썬에서는 "딕셔너리(dictionary)" 또는 **`dict`**로 알려져 있습니다. 파이썬 딕셔너리는 중괄호**`{}`**를 이용하고 **`키 : 값`**의 형태로 각각 나타냅니다.

*.items()

dictionary 내에 있는 데이터를 item() 함수를 통해 (key,value) 이런 튜플 형태로 list 에 넣은 형태로 만들 수 있다.

이렇게 데이터 내부에 자체적인 서브 구조를 가지는 데이터를 구조화된 데이터라고 하겠습니다. 이런 데이터는 나중에 살펴보겠지만, 테이블(table) 형태로 전개됩니다. 위의 treasure_box 데이터는 5개의 행(row), 2개의 열(column)을 가진 데이터가 될 것입니다.

## Pandas

---

pandas라는 파이썬 라이브러리는 Series와 DataFrame이라는 자료 구조를 제공해요. 이 데이터 타입을 활용하면 구조화된 데이터를 더 쉽게 다룰 수 있습니다.

pandas의 특징을 나열하면 다음과 같습니다.

- NumPy기반에서 개발되어 NumPy를 사용하는 어플리케이션에서 쉽게 사용 가능
- 축의 이름에 따라 데이터를 정렬할 수 있는 자료 구조
- 다양한 방식으로 인덱스(index)하여 데이터를 다룰 수 있는 기능
- 통합된 시계열 기능과 시계열 데이터와 비시계열 데이터를 함께 다룰 수 있는 통합 자료 구조
- 누락된 데이터 처리 기능
- 데이터베이스처럼 데이터를 합치고 관계연산을 수행하는 기능

Series에서 인덱스는 기본적으로 정수형태로 설정되고, 사용자가 원하면 값을 할당할 수 있습니다. 따라서 파이썬 딕셔너리 타입의 데이터를 Series 객체로 손쉽게 나타낼 수 있어요.

```python
# 사용자 임의로 index 할당
ser2.index = ['Jhon', 'Steve', 'Jack', 'Bob']

# 조회하면
ser2.index
# Index(['Jhon', 'Steve', 'Jack', 'Bob'], dtype='object')
```

### **Series의 Name**

Series 객체와 Series 인덱스는 모두 name 속성이 있습니다. 이 속성은 pandas의 DataFrame에서 매우 중요해요.

사실 pandas의 DataFrame은 Series의 연속입니다. Series의 name은 DataFrame의 Column명 입니다.

### EDA with Pandas

데이터 분석에 있어서 첫 번째 단계는 무엇일까요? 일단 데이터를 죽 훑어보는 것입니다. 전문 용어로는 **EDA(Exploratory Data Analysis), 우리말로는 데이터를 탐색한다고 표현합니다.**

**.describe()**

이전 스텝에서 학습했던 기본적 통계 데이터(평균, 표준편차 등)를 pandas에서 손쉽게 보고 싶으면 **`.describe()`**을 이용하면 됩니다. **`.describe()`**은 각 컬럼별로 기본 통계데이터를 보여주는 메소드에요. 개수(Count), 평균(mean), 표준편차(std), 최솟값(min), 4분위수(25%, 50% 75%), 최댓값(max)를 보여 줍니다.

**.isnull().sum()**

데이터를 분석할 때, 결측값(Missing value) 확인은 정말 중요한데요. pandas에서 missing 데이터를 isnull()로 확인하고, sum()메소드를 활용해서 missing 데이터 개수의 총합을 구할 수 있습니다.

**`.value_counts()`**

범주형 데이터로 기재되는 컬럼에 대해서는 **`value_counts()`** 메소드를 사용해 각 범주(Case, Category)별로 값이 몇 개 있는지 구할 수 있어요.

**`.value_counts().sum()`**

**`sum()`** 메소드를 추가해서 컬럼별 통계수치의 합을 확인할 수 있습니다.

```python
data['RegionName'].value_counts().sum()
```

**`.corr()`**

data.corr()를 통해 모든 컬럼이 다른 컬럼 사이에 가지는 상관관계를 일목요연하게 확인해 볼 수 있습니다. 상관관계 분석은 EDA에서 가장 중요한 단계라고 할 수 있습니다. 이 과정을 거쳐서 불필요한 컬럼을 분석에서 제외하거나 하게 됩니다.

### **pandas 통계 관련 메소드**

- count(): NA를 제외한 수를 반환합니다.
- describe(): 요약통계를 계산합니다.
- min(), max(): 최소, 최댓값을 계산합니다.
- sum(): 합을 계산합니다.
- mean(): 평균을 계산합니다.
- median(): 중앙값을 계산합니다.
- var(): 분산을 계산합니다.
- std(): 표준편차를 계산합니다.
- argmin(), argmax(): 최소, 최댓값을 가지고 있는 값을 반환 합니다.
- idxmin(), idxmax(): 최소, 최댓값을 가지고 있는 인덱스를 반환합니다.
- cumsum(): 누적 합을 계산합니다.
- pct_change(): 퍼센트 변화율을 계산합니다.

---

# [F-7] 당신의 행운의 숫자는? 나만의 n면체 주사위 위젯 만들기

### **학습 목표**

---

- 파이썬 클래스 문법을 익힙니다.
- 파이썬 클래스를 활용해 객체 지향 프로그래밍에 대해 학습합니다.

### 파이썬에서는 모든 것이 객체다?!

---

*import statement

I want to briefly mention the library search path. Python looks in several places when you try to import a module. Specifically, it looks in all the directories defined in sys.path. This is just a list, and you can easily view it or modify it with standard list methods.

sys.path 에 정의되어있는 경로들을 순차적으로 탐색하여 module 을 import 하는 것이다!

### Everything in Python is an object, and almost everything has attributes and methods.

In Python, the definition is looser; some objects have neither attributes nor methods (more on this in [Chapter 3](https://linux.die.net/diveintopython/html/native_data_types/index.html)), and not all objects are subclassable (more on this in [Chapter 5](https://linux.die.net/diveintopython/html/object_oriented_framework/index.html)).

***But everything is an object in the sense that it can be assigned to a variable or passed as an argument to a function (more in this in [Chapter 4](https://linux.die.net/diveintopython/html/power_of_introspection/index.html)).***

This is so important that I'm going to repeat it in case you missed it the first few times: *everything in Python is an object*. Strings are objects. Lists are objects. Functions are objects. Even modules are objects.

파이썬에서 object라 불리우는 것들은 모두 변수에 할당될 수 있고, 함수의 인자로 넘겨질 수 있는 것들이다. 그러므로 파이썬에 나오는 모든 것들은 object이다.

```python
# 변수는 단지 이름일 뿐이에요. = 연산자를 이용해 값을 할당한다는 의미는 값을 복사하는 것이 아니라 데이터가 담긴 객체에 그냥 이름을 붙이는 것입니다. 진짜 객체를 변수명으로 가리켜 참조할 수 있게 하는 것이죠.
# 즉, 아래 코드는 'cat'이 담긴 문자열 타입의 객체를 생성하고 myword라는 변수(객체)에 할당, 문자열 객체를 참조하게 합니다.
myword = 'cat'
myword
```

### 얕은 복사, 깊은 복사

그리고 이렇게 원본 데이터는 그대로 두고, 참조하는 데이터의 id만을 복사하는 것을 얕은 복사 라고 합니다.

그런데 우리가 생각하는 복사는 동일한 **`[1,2,3,4]`** 데이터가 생기는 것이겠죠.

이런 복사를 파이썬을 비롯한 프로그래밍에서는 **깊은 복사** 라고 합니다.

- 얕은 복사 :copy.copy()
- 깊은 복사 : copy.deepcopy()

원칙적으로 얕은 복사는 원본 객체의 주소를 복사하고, 깊은 복사는 원본 객체의 값을 복사한다고 기억하시면 됩니다.

다만 파이썬에서 얕은 복사, 깊은 복사의 기준은 조금 복잡하므로 우선 서로 다르다는 것만 기억하고, 좀 더 공부한 뒤 필요할 때 다시 공식 문서로 공부해 보세요.

> <요약>
파이썬에서는 모든 것(부울, 정수, 실수, 데이터구조(list,tuple,dict,set…), 함수, 프로그램, 모듈)이 객체다.
객체는 상태(state)를 나타내는 속성(attribute)과 동작(behavior)을 나타내는 메소드(method)가 있다.
객체의 속성은 변수로 구현된다. 객체의 메소드는 함수로 구현된다.

## 객체 지향 프로그래밍

---

우리가 작성한

```
mycar.dirve()
```

코드는 인터프리터 내부에서는

```
Car.drive(mycar)
```

로 동작합니다.

'self' 라는 단어는 클래스를 인스턴스화 한 인스턴스 객체를 가리킵니다.

메소드를 호출할 때, 우리가 명시적으로 인자를 넣지는 않지만 파이썬 내부적으로는 (=파이썬 인터프리터에서는) 인자 한 개를 사용하고 있고, 그 인자는 파이썬 클래스에 의해 선언된 객체 자신(self)입니다.

### '__init__' method

---

클래스를 만들 때부터 색깔과 카테고리를 지정해 주고 싶다면 어떻게 그 값을 전달해 줄 수 있을까요? **init** 을 사용해서 만들 수 있습니다.

- 다른 객체 지향 언어를 알고있는 독자라면 생성자라는 말을 들으면 객체 인스턴스화와 초기화 2가지 작업을 생각할 수 있습니다.
- 그러나 파이썬의 생성자는 초기화만 수행 합니다. 그럼 객체 인스턴스화는 누가 할까요? 기억 나시나요? 네, 바로 클래스 사용시 변수 할당을 통해 이루어집니다.

- **`__init__`** 메소드안에 인자를 전달함로써 인스턴스 객체의 속성을 초기화할 수 있습니다 .
- __init__이라고 쓰고, "던더(Double Under) 이닛"이라고 발음합니다.
- 그리고 이 __init__처럼 앞뒤에 언더바(_)가 두개씩 있는 메소드를 매직 메소드 라고 합니다 ⇒ [https://rszalski.github.io/magicmethods/](https://rszalski.github.io/magicmethods/)
- 즉, **`__init__`** 메소드안에 정의된 속성(변수) **`color`**와 **`category`**는 클래스를 인스턴스화 할 때 값을 설정할 수 있습니다.
- 이를 인스턴스 객체의 초기화(initializing instance) 라고 히고, **`__init__`**함수는 생성자(constructor)라고 합니다.
- **`__init__`** 역시 **`def`** 키워드로 정의합니다. 즉, 클래스안의 메소드이므로 **`self`** 문법 잊지 마세요!

---

`__new__(cls, [...)`

`__new__` is the first method to get called in an object's instantiation. It takes the class, then any other arguments that it will pass along to `__init__`. `__new__` is used fairly rarely, but it does have its purposes, particularly when subclassing an immutable type like a tuple or a string. I don't want to go in to too much detail on `__new__` because it's not too useful, but it is covered in great detail [in the Python docs](http://www.python.org/download/releases/2.2/descrintro/#__new__).

`__init__(self, [...)`

The initializer for the class. It gets passed whatever the primary constructor was called with (so, for example, if we called `x = SomeClass(10, 'foo')`, `__init__` would get passed `10` and `'foo'` as arguments. `__init__` is almost universally used in Python class definitions.

`__del__(self)`

If `__new__` and `__init__` formed the constructor of the object, `__del__` is the destructor. It doesn't implement behavior for the statement `del x` (so that code would not translate to `x.__del__()`). Rather, it defines behavior for when an object is garbage collected. It can be quite useful for objects that might require extra cleanup upon deletion, like sockets or file objects. Be careful, however, as there is no guarantee that `__del__` will be executed if the object is still alive when the interpreter exits**, so `__del__` can't serve as a replacement for good coding practices (like always closing a connection when you're done with it. In fact, `__del__` should almost never be used because of the precarious circumstances under which it is called; *use it with caution!***

---

### why?

객체와 객체지향 프로그래밍을 왜 사용할까요? 그 이유를 전문 용어로 설명하면 추상화와 캡슐화를 위함입니다

- 추상화(abstraction) :  복잡한 자료, 모듈, 시스템 등으로부터 핵심적인 개념 또는 기능을 간추려 내는 것을 말한다.
- 캡슐화 : 객체의 속성과 행위를 하나로 묶고 실제 구현 내용 일부를 외부에 감추어 은닉하는 것.

[ex]

Car 클래스는 자동차의 여러 기능 중 운전하기(drive), 가속하기(accel) 기능을 제공합니다. 

각 기능은 추상화하여 함수로 제공하기 때문에 자동차 클래스를 만들어 사용하는 사람은 세부 구현에 대해서는 알지 못해도 필요한 함수를 호출하는 것만으로 손쉽게 해당 기능을 사용할 수 있습니다.

또한 각 기능을 Car 클래스로 캡슐화하여 제공하므로 자동차 클래스가 필요한 사람은 언제나 Car 클래스라는 캡슐을 쉽게 가져다 쓸 수 있습니다.

## 상속

---

### **상속 사용하기**

---

클래스를 잘 상속하기 위해 필요한 3가지 사용법을 살펴 보겠습니다.

- 메소드 추가하기(add)
- 메소드 재정의하기(override)
- 부모 메소드 호출하기(**`super()`**)

이렇게 기존에 있는 메소드를 변경하는 것을 메소드 오버라이드(재정의, override) 라고 합니다.

**`super`**([*type*[, *object-or-type*]])

Return a proxy object that delegates method calls to a parent or sibling class of *type*. This is useful for accessing inherited methods that have been overridden in a class.

super()를 사용하는 이유?

⇒부모 클래스의 변경사항이 그대로 자식 클래스에 반영됩니다.

## 요약

---

1. **클래스 선언**

**2. 클래스 사용**

- "객체의 인스턴스"

**3. 클래스는 동작과 상태를 갖는다.**

- 상태(State): 속성(Attribute)로 표현, 일명 변수
- 동작(Behavior): Methode로 표현, 일명 함수
- **객체는 동작은 공유하지만 상태는 공유하지 않는다.**

**4. 생성자** **`__init__`**

**5. 클래스 변수와 인스턴스 변수**

- 클래스에 선언된 속성은 클래스 변수라고 하며 이 클래스에 의해 생성된 모든 객체에 대해 같은 속성(값)을 갖는다.
- 객체가 인스턴스화 될 때마다 새로운 값이 할당되며 서로 다른 객체 간에는 속성(값)을 공유할 수 없다.

**6. 상속**

- 메소드 추가, 메소드 오버라이드, 부모메소드 호출하기

---

## 실습 : 주사위 만들기

파이썬 클래스의 state 및 method 선언해보고 사용해보기