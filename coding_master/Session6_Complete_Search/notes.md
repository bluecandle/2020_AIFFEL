[참조 1](https://brenden.tistory.com/10)
[참조 2](https://developer-mac.tistory.com/81)

## 정의
가능한 경우의 수를 일일이 나열하면서 답을 찾는 방법.
가능한 방법을 전부 만들어 보는 알고리즘
Exhaustive Search

### 방법들
- Brute Force : for 문과 if 문을 이용하여 처음부터 끝까지 탐색하는 방법
- 비트마스크 : 이진수 표현을 자료구조로 쓰는 기법 (AND, OR, XOR, SHIFT, NOT)
- 순열 : 순열의 시간 복잡도는 O(N!)
    서로 다른 n개의 원소에서 r개의 중복을 허용하지 않고 순서대로 늘어 놓은 수
- [백트래킹](https://semtax.tistory.com/50)
- [DFS](https://www.notion.so/bluecandle/DFS-Depth-First-Search-880f37d582904d828f2222488ce19c9d)
- [BFS](https://www.notion.so/bluecandle/BFS-Breadth-First-Search-0ee42b081d85410286c458afa355939d)


정말 아무 방법이 없어보이는 답이 없는 문제가 의외로 문제 크기가 작아서 진짜 일일이 다 시도해보는 게 가능할 때도 있다!


https://www.acmicpc.net/problem/2098
=> 유명한 NP문제이다. 풀이법이 상당히 까다로운데 순열문제를 활용하면 된다. 그리고 이 문제는 DFS도 연습할 수 있으므로 2가지 방법으로 풀어보는 것이 좋다.

*추가, BFS 복습한김에 : [다익스트라](https://www.notion.so/bluecandle/04d3a93bef764b788c4b2414436feefd)