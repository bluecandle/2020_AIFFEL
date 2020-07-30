# 0 또는 양의 정수가 주어졌을 때, 정수를 이어 붙여 만들 수 있는 가장 큰 수를 알아내 주세요.

# 예를 들어, 주어진 정수가 [6, 10, 2]라면 [6102, 6210, 1062, 1026, 2610, 2106]를 만들 수 있고, 이중 가장 큰 수는 6210입니다.

# 0 또는 양의 정수가 담긴 배열 numbers가 매개변수로 주어질 때, 순서를 재배치하여 만들 수 있는 가장 큰 수를 문자열로 바꾸어 return 하도록 solution 함수를 작성해주세요.

# 제한 사항
# numbers의 길이는 1 이상 100,000 이하입니다.
# numbers의 원소는 0 이상 1,000 이하입니다.
# 정답이 너무 클 수 있으니 문자열로 바꾸어 return 합니다.

# 입출력 예
# numbers	return
# [6, 10, 2]	6210
# [3, 30, 34, 5, 9]

numbers = [6, 10, 2]
# numbers = [6, 10, 2,100,219, 200,201,90,91,210]
def solution(numbers):
    
    # 한 자리, 두 자리, 세 자리를 나눈다.
    # 0~9
    # class1 = []
    # # [][]
    # class2 = [[]*1 for i in range(10)]
    # # [][][]
    # class3 = [[[]*1 for i in range(10)]*1 for j in range(10)]    
    # class4 = []

    # # 입력된 모든 숫자를 분류한다. O(N)
    # for number in numbers:
    #     if(number <10):
    #         class1.append(number)
    #     elif(number <100):
    #         class2[(number//10)].append(number)
    #     elif(number<1000):
    #         class3[(number//100)][(number//10)%10].append(number)
    #     else:
    #         class4.append(number)

    # # 입력된 행렬들을 각각 정렬. O(1)
    # class1.sort()
    
    # for i in range(len(class2)):
    #     class2[i].sort()
    
    # for i in range(len(class3)):
    #     for j in range(len(class3[i])):
    #         class3[i][j].sort()
    
    # print(class1,class2,class3)

    
    
    answer = ''
    return answer

print(solution(numbers))