# AIFFEL_21일차 2020.08.21

Tags: AIFFEL_DAILY

### 일정

---

1. LMS E8 마무리
2. LMS F18 하기
3. cs231n lecture07 TNN2 풀잎

# [F-18]파이썬을 포토샵처럼 쓸 수 있다고?

---

## **학습 목표**

---

- PIL 패키지를 이용한 이미지 처리
- 함수와 모듈 제작 및 실습

RGBA 색 공간 범위 : 우리가 자주 쓰는 색 공간은 RGB 혹은 RGBA 입니다.

A는 Alpha 입니다. Alpha는 투명도를 의미하며, 0이면 해당 픽셀이 투명해집니다.

### crop

```python
region = im.crop(box)
```

### resize

```python
resized_image = im.resize((100,200))
```

- 이미지를 자르지 않고 사이즈를 줄이는 방법은 resize()를 사용하는 것입니다. im.resize((100, 200))와 같이 resize 안에 바꾸고 싶은 사이즈를 입력합니다. 여기서는 가로 100, 세로 200으로 사이즈를 바꿨습니다.

### rotate

```python
region.transpose(Image.ROTATE_180)

# 혹은

region = region.rotate(180)
```

이 코드가 이미지를 회전시키는 기능을 합니다.

### 합치기

```python
im.paste(region, box)
im.show()

# 두 개의 이미지를 합치면 자르고 회전시킨 이미지(region)이
# 원본 이미지(im) 중간에 들어간 구조가 되게 됩니다.
```

### 대비(contrast) 변경

---

이미지 대비는 **물체를 다른 물체 또는 배경과 구별할 수 있도록 만들어 주는 시각적인 특성차**를 말합니다. 즉, **대비**는 한 **물체와 다른 물체의 색과 밝기의 차이**로 결정됩니다.

```python
from PIL import Image
from PIL import ImageEnhance

im = Image.open(img_path)                # original 이미지

enh = ImageEnhance.Contrast(im)    # enhanced contrast 이미지
enh = enh.enhance(1.9)
```

### 이미지 필터

---

```python
from PIL import ImageFilter

filtered_image = im.filter(ImageFilter.BLUR)
```

필터 종류

- BLUR : 이미지를 흐리게 만드는 필터입니다.
- EDGE_ENHANCE : 윤곽을 뚜렷하게 해주는 필터입니다.
- EMBOSS : 원본 이미지의 명암 경계에 따라 이미지의 각 픽셀을 밝은 영역 또는 어두운 영역으로 대체하는 필터입니다.
- FIND_EDGES : 윤곽만 표시해주는 필터입니다.
- SHARPEN : 경계선들을 더욱 날카롭게 해주어 선명도를 높이는 필터입니다.

이미지 필터 목록 : [https://pillow.readthedocs.io/en/latest/reference/ImageFilter.html](https://pillow.readthedocs.io/en/latest/reference/ImageFilter.html)

### 색 공간 변경

---

```python
greyscale_image = im.convert('L')
greyscale_image.show()
```

```python
from PIL import Image, ImageFilter

def image_resize(image, height):
    if height == 300:
        return image.resize((800, 300))
    else :
        return image.resize((800, 600))

def image_rotate(image):
    return image.transpose(Image.ROTATE_180)

def image_change_bw(image):
    return image.convert('L')
```

```python
from PIL import Image
import os
import sys
sys.path.append(os.getenv('HOME')+'/aiffel/pil_image')    # 우리가 추가한 모듈의 path를 sys.path에 임시로 추가해 줍니다. 

import image_processing as ip      # 위 추가한 path에서 image_processing.py 모듈을 가져와 임포트합니다. 

original_img_path = os.getenv('HOME')+'/aiffel/pil_image/assets/test.jpg'
result_img_path = os.getenv('HOME')+'/aiffel/pil_image/assets/result_image.jpg'

def img_transfer(original_image, result_image):
    # 원본 이미지를 오픈합니다. 
    im = Image.open(original_image)

    # image_processing.image_resize 를 사용하여 (800,600)으로 resize합니다. 
    im_resized_600 = ip.image_resize(im, 600)
    # image_processing.image_resize 를 사용하여 (800,300)으로 resize합니다. 
    im_resized_300 = ip.image_resize(im, 300)

    # image_processing.image_rotate 를 사용하여 (800,300)짜리 이미지를 180도 회전합니다. 
    im_resized_300_rotate = ip.image_rotate(im_resized_300)

    # im_resized_600에 im_resized_300와 im_resized_300_rotate를 아래위로 붙입니다. 
    box_top = (0, 0, 800, 300)
    box_bottom = (0, 300, 800, 600)
    im_resized_600.paste(im_resized_300_rotate, box_top)
    im_resized_600.paste(im_resized_300, box_bottom)

    # image_processing.image_change_bw 를 사용하여 im_resized_600을 흑백으로 변환합니다.
    im_resized_600 = ip.image_change_bw(im_resized_600)

    # 이미지를 저장합니다. 
    im_resized_600.save(result_image)

# 함수를 호출해서 원본이미지를 새로운 이미지로 변환해서 저장해 봅시다.     
img_transfer(original_img_path, result_img_path)

# 만들어진 이미지를 화면에 출력해 봅시다. 
Image.open(result_img_path).show()
```