# max_time = 0.09 ms
from collections import defaultdict
import heapq
import operator
def solution(genres, plays):
    play_genre = defaultdict(int)
    best_play = defaultdict(list)

    for idx, val in enumerate(genres):
        p = plays[idx]
        play_genre[val] += p
        heapq.heappush(best_play[val], [-p, idx])

    answer = []
    for item in sorted(play_genre.items(), key = operator.itemgetter(1), reverse = True):
        genre = item[0]
        answer += [heapq.heappop(best_play[genre])[1]]
        if best_play[genre]:
            answer += [heapq.heappop(best_play[genre])[1]]

    return answer