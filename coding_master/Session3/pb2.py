# 문제 설명
# 전화번호부에 적힌 전화번호 중, 한 번호가 다른 번호의 접두어인 경우가 있는지 확인하려 합니다.
# 전화번호가 다음과 같을 경우, 구조대 전화번호는 영석이의 전화번호의 접두사입니다.

# 구조대 : 119
# 박준영 : 97 674 223
# 지영석 : 11 9552 4421
# 전화번호부에 적힌 전화번호를 담은 배열 phone_book 이 solution 함수의 매개변수로 주어질 때,
# 어떤 번호가 다른 번호의 접두어인 경우가 있으면 false를 그렇지 않으면 true를 return 하도록 solution 함수를 작성해주세요.

# 제한 사항
# phone_book의 길이는 1 이상 1,000,000 이하입니다.
# 각 전화번호의 길이는 1 이상 20 이하입니다.
# 입출력 예제
# phone_book	return
# [119, 97674223, 1195524421]	false
# [123,456,789]	true
# [12,123,1235,567,88]	false
# 입출력 예 설명
# 입출력 예 #1
# 앞에서 설명한 예와 같습니다.

# 입출력 예 #2
# 한 번호가 다른 번호의 접두사인 경우가 없으므로, 답은 true입니다.

# 입출력 예 #3
# 첫 번째 전화번호, “12”가 두 번째 전화번호 “123”의 접두사입니다. 따라서 답은 false입니다.

def solution(phone_book):

    ## 이렇게하면 테스트 케이스는 다 넘어가지만, 시간초과 나옴
    
    # answer = True
    # phone_book = sorted(phone_book,key=lambda x: len(x))
    
    # for idx,p in enumerate(phone_book):
    #     for i in range(idx+1,len(phone_book)):
    #         b = phone_book[i][:len(p)]
    #         if(p == b):
    #             answer = False
    #             break   

    # return answer    
    
    answer = True
    # 생각해보니, phone_book 에 들어있는 숫자들을 길이에 따라 sort 할 필요 없이, 그냥 sort 하면 최대한 비슷한 숫자들끼리 붙여서 정렬을 해주게된다.
    # string 비교의 특성에 의해!
    phone_book.sort()

    # 이렇게 key 라는 값에 lambda 사용하듯(lambda도 함수! 익명함수) 함수를 사용할 수 있다.   
    # key_test = sorted(phone_book, key=len)
    # print(key_test)
                
    for idx in range(len(phone_book)-1): 

               
        p1 = phone_book[idx]
        p2 = phone_book[idx+1]

        if p1 in p2:
            answer = False
            break

        

    return answer 
    

  
print(solution(['119', '97674223', '1195524421']))
print(solution(['123','456','789']))
print(solution(['123','456','789','123']))

