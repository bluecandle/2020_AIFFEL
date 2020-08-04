import collections


def solution_1(participant, completion):
    answer = collections.Counter(participant) - collections.Counter(completion)
    return list(answer.keys())[0]

def solution_2(participant, completion):
    answer = ''
    temp = 0
    dic = {}
    for part in participant:
        dic[hash(part)] = part
        temp += int(hash(part))
    for com in completion:
        temp -= hash(com)
    # temp 변수를 활용하여 hash 값을 활용할 수 있도록 만들었음
    answer = dic[temp]

    return answer


# solution(['leo', 'kiki', 'eden'], ['kiki', 'eden'])
print(solution_1(['leo', 'kiki', 'eden'], ['kiki', 'eden']))
print(solution_2(['mislav', 'stanko', 'mislav', 'ana'], ['stanko','ana', 'mislav']))