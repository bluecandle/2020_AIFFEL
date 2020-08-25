from flask import Flask

app = Flask(__name__)

from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print ('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result
    return wrap

@app.route("/")
def hello():    
    return "플라스크 동작 확인!"

if __name__ == "__main__":
    app.run()



