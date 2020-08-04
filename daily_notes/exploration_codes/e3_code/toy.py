import time
import sys

width = 101

for i in range(width):
    time.sleep(0.001)
    if i % 10 == 0:
        sys.stdout.write('')
        sys.stdout.flush()
    else:
        sys.stdout.write(".")
        sys.stdout.flush()

sys.stdout.write("\n")