import tensorflow as tf

class ScatteringTensorFlow(tf.Module):
    def __init__(self, name):
        super(ScatteringTensorFlow, self).__init__(name=name)
        self.frontend_name = 'tensorflow'

    @tf.Module.with_name_scope
    def __call__(self, x):
        """This method is an alias for `scattering`."""
        return self.scattering(x)

    _doc_array = 'tf.Tensor'
    _doc_array_n = ''

    _doc_alias_name = '__call__'

    _doc_alias_call = ''

    _doc_frontend_paragraph = r"""

        This class inherits from `tf.Module`. As a result, it has all the
        same capabilities as a standard TensorFlow `Module`."""

    _doc_sample = 'np.random.randn({shape})'

    _doc_has_shape = True

    _doc_has_out_type = True
