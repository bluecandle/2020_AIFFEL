import math
def solution(progresses, speeds):
    answer = []
    # 완료하는데 걸리는 날짜로 변환
    progresses = [math.ceil((100-p)/s) for p,s in zip(progresses, speeds)]
    
    flag_idx = 0
    for idx in range(len(progresses)):
        # 기점(flag)보다 더 많은 시간이 걸리는 작업이 나올 때까지 찾는다.
        if progresses[flag_idx] < progresses[idx]:            
            # 찾으면, 나온 지점과 flag 지점의 차이만큼을 구해서 답에 넣어준다.        
            answer.append(idx-flag_idx)
            # flag 갱신
            flag_idx = idx

    # 갱신된 flag_idx 에 들어있는 값이 해당 index 이후에 들어있는 값들보다 크다면,
    # ex, [7,3,9,8,7]
    # 해당 idx에서부터 나머지 작업의 갯수를 다 합한 값이 마지막에 들어가게 된다.
    # 이 부분을 제대로 처리하지않아서 시도1,2에서 틀렸던 것으로 생각됨.
    answer.append(len(progresses)-flag_idx)

    return answer

print(solution([93,30,55],[1,30,5]))
print(solution([93,30,55,60,65],[1,30,5,5,5]))