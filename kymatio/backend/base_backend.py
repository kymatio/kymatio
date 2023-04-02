## Basic common functionality for all backends
## Ideas taken from tensorly https://github.com/tensorly/tensorly/blob/main/tensorly/backend/core.py



backend_types = [
    "int32",
    "int64",
    "float32",
    "float64",
    "complex64",
    "complex128",
    "pi",
    "e",
    "inf",
]
backend_basic_math = [
    "exp",
    "log",
    "tanh",
    "cosh",
    "sinh",
    "sin",
    "cos",
    "tan",
    "arctanh",
    "arccosh",
    "arcsinh",
    "arctan",
    "arccos",
    "arcsin",
]
backend_array = [
    "einsum",
    "matmul",
    "ones",
    "zeros",
    "any",
    "prod",
    "all",
    "where",
    "reshape",
    "cumsum",
    "count_nonzero",
    "eye",
    "sqrt",
    "abs",
    "min",
    "zeros_like",
]



class BaseBackend:
    pass
