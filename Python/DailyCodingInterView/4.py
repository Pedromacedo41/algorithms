
import numpy as np
from collections import  deque

class MovingAverage:
    def __init__(self, size: int):
        self.size: int = size
        self.queue = deque(maxlen=size)


    def next(self, val: int) -> float:
        self.queue.append(val)
        return np.mean(list(self.queue))
        

if __name__ == "__main__":
     m = MovingAverage(1)
     print(m.next(4))
     print(m.next(0))


    