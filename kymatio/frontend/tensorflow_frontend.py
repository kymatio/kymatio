class ScatteringTensorflow(object):
   def __init__(self, J, shape, max_order = None):
       super(ScatteringTensorflow, self).__init__()
       self.J = J
       self.shape = shape
       self.max_order = max_order