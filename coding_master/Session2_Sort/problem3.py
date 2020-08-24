# 문제 설명
# H-Index는 과학자의 생산성과 영향력을 나타내는 지표입니다. 어느 과학자의 H-Index를 나타내는 값인 h를 구하려고 합니다. 위키백과1에 따르면, H-Index는 다음과 같이 구합니다.

# 어떤 과학자가 발표한 논문 n편 중, h번 이상 인용된 논문이 h편 이상이고 나머지 논문이 h번 이하 인용되었다면 h의 최댓값이 이 과학자의 H-Index입니다.

# 어떤 과학자가 발표한 논문의 인용 횟수를 담은 배열 citations가 매개변수로 주어질 때, 이 과학자의 H-Index를 return 하도록 solution 함수를 작성해주세요.

# 제한사항
# 과학자가 발표한 논문의 수는 1편 이상 1,000편 이하입니다.
# 논문별 인용 횟수는 0회 이상 10,000회 이하입니다.
# 입출력 예
# citations	return
# [3, 0, 6, 1, 5]	3
# 입출력 예 설명
# 이 과학자가 발표한 논문의 수는 5편이고, 그중 3편의 논문은 3회 이상 인용되었습니다. 그리고 나머지 2편의 논문은 3회 이하 인용되었기 때문에 이 과학자의 H-Index는 3입니다.

# citation 숫자 배열 안에서 가장 많이 인용된 횟수부터 시작하여, 순차적으로 1씩 감소시켜보며 진행한다.
# h가 0이 될 때까지 진행하며,h 가 0이 되는 경우 답은 0이 된다. ( 모든 논문이 한 번도 인용된 적이 없음)
# h 번 이상 인용된 논문들을 골라내고, 골라진 논문들의 수가 h 개 이상이라면, 우선 1단계 통과
# 단, 골라진 논문들을 내림차순으로 정렬했을 때 h 번째 이후에 있는 대상들을 확인해봐야한다.
# 해당 대상들이 h 번을 초과하는 인용 횟수를 가졌다면, 조건에 성립하지 않으므로, 다음 h 항목으로 넘어가야 한다.

def solution(citations):
    answer=0   
    for h in range(max(citations),0,-1):
        # print(h)
        _l = sorted(list(filter(lambda x: x>=h, citations)),reverse=True)
        # print(_l)
        if(len(_l)>=h):
            __l = _l[h:]
            # print(__l)
            answer = h
            for o in __l:
                if(o>h):
                    answer = 0
                    break
            if(answer):
                break

    return answer
# print(solution([3, 0, 6, 1, 5]))
# print(solution([4,4,2,7]))
# print(solution([0,0,0,0]))