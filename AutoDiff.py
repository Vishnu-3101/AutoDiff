import numpy as np
import sys

class Tensor:
    '''
    Converts each Tensor into a class of type Tensor.
    Operator overloading happens for each operator.
    exp: 
        x1+x2 => x1 is self
                 x2 is other
    After every operation, a new result node is created with operand nodes set to prev. This helps during backpropagation from destination node to source node to calculate gradients.
    '''
    def __init__(self,data, _prev=(),_op=''):
        '''
        Args:
            data: Holds the data of the node
            grad: The gradient Tensor
            _backward: Function to calculate gradients of prev nodes
                    => This is called in the one of the prev node operation which is responsible to create the current node.
            _prev: Keeps track of the previous nodes that generated the current node
            _op: Holds the operation performed on prev nodes to generate current node
        '''
        self.data = data
        self.grad = 0
        self._backward = lambda : None
        self._prev = set(_prev)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self,other):
        other = other if isinstance(other,Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self,other),'+')
        def _backward():
            '''This function is executed only when _backward is called explictly or when node._backward() is called. When node._add__ is called, this function don't get executed. It will only be initialized to node._backward = 'function address'
            '''
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    
    def __mul__(self,other):
        other = other if isinstance(other,Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self,other),'*')
        def _backward():
            self.grad += out.grad*other.data
            other.grad += out.grad*self.data
        out._backward = _backward
        return out
    

    '''The __radd__ and __rmul__ represents reverse addition and multiplication
    Ex: 5+x1, 5 of type int cannot be added with a Tensor class.
        So in such cases __radd__ is called which converts 5+x1 to x1+5.
    '''
    def __radd__(self,other):
        return self+other
    
    def __rmul__(self,other):
        return self+other
    
    def backward(self):
        '''Perfrom topological sort using DFS.
        For every directed edge u-v, vertex u comes before v in the ordering.
        '''
        visited = set()
        topo = []

        def build_grad(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_grad(child)
                topo.append(node)

        '''The gradient of output node is set to 1. 
        Since backward is called only once, it is declared here..
        '''
        self.grad = 1             
        build_grad(self)
        for node in reversed(topo):
            node._backward()