# AIFFEL 3일차 2020.07.27

Tags: AIFFEL_DAILY

# CodingMaster 1회차

### List 수업 같이 듣기, 문제 풀어보기

[https://swexpertacademy.com/main/learn/course/subjectDetail.do?courseId=AVuPDN86AAXw5UW6&subjectId=AWOVFCzaqeUDFAWg&&](https://swexpertacademy.com/main/learn/course/subjectDetail.do?courseId=AVuPDN86AAXw5UW6&subjectId=AWOVFCzaqeUDFAWg&&)

### 개념 정리 및 문제 풀이

[https://github.com/bluecandle/2020_AIFFEL/tree/master/coding_master/Session1](https://github.com/bluecandle/2020_AIFFEL/tree/master/coding_master/Session1)

# LMS 수업

## [F-1] Intro

### tensorflow 2.0

API 정리, 즉시 실행, 전역 메커니즘 제거, 세션 대신 함수

가장 중요한 흐름은 훨씬 직관적이면서도 편리해졌다는 점입니다. 여러 가지로 분산되어 있던 API를 통합해 일관성을 키우고, 복잡한 코드가 필요했던 부분을 제거하여 깔끔하게 사용할 수 있게 하는 등 사용자를 위한 변화가 돋보입니다.

### 터미널 사용법 익히기

다만, 여기서 한 가지 주의할 점은 cp 명령어는 rm(삭제) 명령어와 같이 디렉토리를 복사할 때 -r 옵션을 추가해주어야 복사하려는 디렉토리의 하위디렉토리까지 함께 복사한다는 점입니다. 개별 파일을 복사하고 싶을 때에는 -r 이 필요없죠.

**패키지(Package)** 란, 간단히 말해서 특정 기능을 하는 작은 프로그램 단위입니다. 위에서 설명한 라이브러리와도 일맥상통하는 개념이죠. 패키지가 조금 더 큰 범위를 포괄한다고 볼 수도 있지만, 두 용어를 같은 의미로 쓰기도 합니다.

개발을 하다보면 다양한 패키지들이 필요합니다. 패키지는 우리가 다운 받아 사용할 수 있는 다양한 툴을 제공하기 때문에, 효율적인 개발 작업을 위해서는 많은 패키지들을 설치하거나 삭제하는 등 자유자재로 관리할 수 있어야 합니다. 우분투에서 패키지를 관리하기 위해 주로 쓰이는 명령어는 **`apt-get`** 입니다.

*sudo

본래 "superuser do"에서 유래하였으나, 후에 프로그램의 기능이 확장되며 "substitute user do"(다른 사용자의 권한으로 실행)의 줄임말로 해석되게 되었다.

```
$ sudo apt-get upgrade
```

⇒ 모든 패키지에 대해, 새롭게 업데이트 된 버전이 있다면 전부 업그레이드를 하는 명령어입니다. 하지만 패키지를 최신 버전으로 업그레이드 하는 것은 언제나 기존에 함께 사용되던 패키지들과의 충돌을 야기할 수 있기 때문에, 주의해야 합니다.

-y 라는 옵션은 설치 중간중간 나오는 질문들에 대해 모두 yes 로 답하겠다는 옵션입니다. -y 를 빼고 입력해서 중간 중간 직접 yes 를 입력해줘도 됩니다.

### 파이썬 가상환경

pip install -r requirements.txt 라고 했을 때 -r 이 —requirement 라는 의미

아나콘다는 전용 가상환경을 제공하기 때문에, 해당 환경을 사용하는 것을 권장한다.

conda 는 venv와 달리 가상 환경을 현재 경로에 생성하지 않고 아나콘다 설치 폴더의 envs 안에 생성한다.

## [F-3] 개발환경 익숙해지기

### Git

git repo 의 구조는 크게 세가지로 구성되어 있다.

작업폴더, 인덱스(staging area), 저장소

![Untitled.png](Untitled.png)

[출처]([https://ifuwanna.tistory.com/193](https://ifuwanna.tistory.com/193))

### Jupyter Notebook

Jupyter Notebook은 위와 같이 "문서" 작업과 "코드" 작업을 동시에 진행할 수 있는 어플리케이션입니다.

가상환경에는 우리가 필요로 하는 텐서플로우 등 다양한 라이브러리가 설치될 것입니다.

그래서 우리는 방금 만들어진 가상환경을 jupyter notebook이 코드 구동을 위해 사용할 수 있는 파이썬 환경인 커널로 등록해보겠습니다. 그러기 위해서는

```
ipykernel
```

이라는 모듈을 추가로 설치해 주어야 합니다.

다음으로 셀의 타입으로는 위와 같이 두 가지가 있습니다. 마크다운 셀은 제목 또는 설명 등을 입력하고, 코드 셀은 실행시킬 수 있는 파이썬 코드를 입력합니다.

특정 셀을 마크다운 셀 또는 코드 셀로 변환하고 싶다면 다음과 같은 단축키를 사용합니다.

- 마크다운 셀로 변환하기 : **`esc`** + **`m`** (명령모드로 변환 후 **`m`** )
- 코드 셀로 변환하기 : **`esc`** + **`y`** (명령모드로 변환 후 **`y`** )

이 외에 가장 많이 쓰이는 노트북 단축키는 다음과 같습니다. 단축키는 정말 살이 되고 피가 되니, 한 번씩 누르면서 익혀보세요!

- 셀의 실행 : **`Shift`** + **`Enter`**
- 셀 삭제 : **`esc`** + **`x`** or **`esc`** + **`dd`**
- 셀 삭제 취소 : **`esc`** + **`z`**
- 위에 셀 추가 : **`esc`** + **`a`**
- 아래에 셀 추가 : **`esc`** + **`b`**

이미 눈치채셨겠지만, 단축키 사용 시 가장 많이 쓰이는 키는 바로 **`esc`** 키입니다. 특정 행동을 하려고 할 때 주로 명령 모드에서 많이 진행하기 때문이죠.

역시, **`esc`** 를 잊으면 안되겠죠?

이 외에도 주피터 노트북에는 단축키가 많은데요, 다른 단축키가 궁금하다면 명령모드에서 **`H`** 를 눌러보세요.

### Markdown

줄바꿈

마크다운에서는 그냥 enter로는 줄바꿈이 되지 않습니다. 문장의 맨 끝에 띄어쓰기를 세 번 이상 해야 가능하죠.

### CS231n lecture03

[CS231n 2017 lecture3](https://www.notion.so/CS231n-2017-lecture3-a167e077f2ea4a8aaf59fd0a508013cf)
