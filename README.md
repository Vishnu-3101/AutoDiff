# AutoGrad

An implementation of Automatic Differentiation from scratch.

- This helps understand how differentiation of complex functions is performed in libraries like PyTorch and Tensorflow.
- It uses the concepts of Operator overloading and Topological sort.

`The current implementation supports only multiplication and addition operators. More operators will be added further.`

## Steps to run the code

1. Clone the directory
2. Install required libraries

```
pip install -r requirements.txt
```
3. Replace the function in main.py with the function you want to calculate gradient for
4. Run the main file to git the gradients and the graph flow.
```
python main.py
```