# funnydice.py

from random import randrange

class FunnyDice:
    def __init__(self, n=6):
        self.n = n
        self.options = list(range(1, n+1))
        self.index = randrange(0, self.n)
        self.val = self.options[self.index]
    
    def throw(self):
        self.index = randrange(0, self.n)
        self.val = self.options[self.index]
    
    def gettingval(self):
        return self.val
    
    def settingval(self, val):
        if val <= self.n:
            self.val = val
        else:
            print("주사위에 없는 숫자입니다. 주사위는 1 ~ {0}까지 있습니다. ".format(self.n))
            raise error

def get_inputs():
    n = int(input("주사위 면의 개수를 입력하세요: "))
    return n

def main():
    n = get_inputs()
    mydice = FunnyDice(n)
    mydice.throw()
    print("행운의 숫자는? {0}".format(mydice.gettingval()))

if __name__ == '__main__':
    main()