from AutoDiff import Tensor
from utils import DrawGraph2

x1 = Tensor(2)
x2 = Tensor(3)

def SimpleFunction(x1,x2):
    return (x1*x2)*(x1+x2)

y = SimpleFunction(x1,x2)
y.backward()

DrawGraph2(y)