from .numpy_frontend import ScatteringNumPy


class ScatteringJax(ScatteringNumPy):

    def __init__(self):
        self.frontend_name = 'jax'
