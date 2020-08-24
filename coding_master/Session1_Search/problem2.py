# 기본 제공코드는 임의 수정해도 관계 없습니다. 단, 입출력 포맷 주의
# 아래 표준 입출력 예제 필요시 참고하세요.

# 표준 입력 예제
'''
a = int(input())                        정수형 변수 1개 입력 받는 예제
b, c = map(int, input().split())        정수형 변수 2개 입력 받는 예제 
d = float(input())                      실수형 변수 1개 입력 받는 예제
e, f, g = map(float, input().split())   실수형 변수 3개 입력 받는 예제
h = input()                             문자열 변수 1개 입력 받는 예제
'''

# 표준 출력 예제
'''
a, b = 6, 3
c, d, e = 1.0, 2.5, 3.4
f = "ABC"
print(a)                                정수형 변수 1개 출력하는 예제
print(b, end = " ")                     줄바꿈 하지 않고 정수형 변수와 공백을 출력하는 예제
print(c, d, e)                          실수형 변수 3개 출력하는 예제
print(f)                                문자열 1개 출력하는 예제
'''

import sys


'''
      아래의 구문은 input.txt 를 read only 형식으로 연 후,
      앞으로 표준 입력(키보드) 대신 input.txt 파일로부터 읽어오겠다는 의미의 코드입니다.
      여러분이 작성한 코드를 테스트 할 때, 편의를 위해서 input.txt에 입력을 저장한 후,
      아래 구문을 이용하면 이후 입력을 수행할 때 표준 입력 대신 파일로부터 입력을 받아올 수 있습니다.

      따라서 테스트를 수행할 때에는 아래 주석을 지우고 이 구문을 사용하셔도 좋습니다.
      아래 구문을 사용하기 위해서는 import sys가 필요합니다.

      단, 채점을 위해 코드를 제출하실 때에는 반드시 아래 구문을 지우거나 주석 처리 하셔야 합니다.
'''
sys.stdin = open("input2.txt", "r")

# K 충전 후 한번에 이동 가능한 정류장 수
# N 이 종점
# M 충전기가 설치된 정류장 번호 수
# 도착할 수 없으면 0
T = int(input())

# 여러개의 테스트 케이스가 주어지므로, 각각을 처리합니다.
for test_case in range(1, T + 1):
    
    # 최소 몇 번의 충전을 통해 종점에 도착할 수 있느냐??
    steps = 0
    K, N, M = map(int, input().split())
    stations =list(map(int,input().split()))
    # print(K,N,M)
    # print(stations)
    # N 까지 가야하는데, 한 번 충전 이후에 K 개의 정류장을 이동할 수 있고, stations list 에 충전기가 설치된 정류장의 번호가 있다.

    isPossible = True    

    # [1] 불가능한 경우 거르기
    for i in range(0,len(stations)-1):
        prev = i
        to = i+1
        diff = stations[to] - stations[prev]
        if(diff > K):
            isPossible = False
            break

    # [1] 불가능한 경우 거르기
    if(not isPossible):
        print('#{0} {1}'.format(test_case,steps))
    
    # [2] 가능한 경우, step 을 구한다
    else:
        steps = 0
        currentNum = 0
        prevNum = 0
        while(currentNum<N):
            # 우선 최대한 갈 수 있는 만큼 이동
            currentNum += K
       
            if(currentNum>=N):               
                break

            # 갈 수 있는 만큼 이동한 곳에 정류소가 있으면 step 증가
            if(currentNum in stations):               
                steps+=1

            # 존재하지 않음 => 가장 가까운 정류소를 찾는다
            else:
                # 출발지점과 일단 최대로 이동한 지점 사이에서 최대로 이동한 지점과 가장 가까운 station 을 찾는다.
                sts = [x for x in stations if x > prevNum and x< currentNum]
                # print(sts)           
                station_to = min(sts, key = lambda x: currentNum-x)
                # cands = next(x for x in stations if (x > prevNum and min(currentNumx)))
                # print('asdf',station_to)
                currentNum = station_to            
                steps+=1

            prevNum = currentNum

        print('#{0} {1}'.format(test_case,steps))            


    # ///////////////////////////////////////////////////////////////////////////////////
