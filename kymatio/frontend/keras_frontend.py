from tensorflow.keras.layers import Layer


class ScatteringKeras(Layer):
    def __init__(self):
        Layer.__init__(self)
        self.frontend_name = 'keras'

    def build(self, input_shape):
        self.shape = input_shape
        Layer.build(self, input_shape)

    def scattering(self, x):
        return self.S.scattering(x)

    def call(self, x):
        return self.scattering(x)

    _doc_array = 'tf.Tensor'
    _doc_array_n = ''

    _doc_alias_name = 'call'

    _doc_alias_call = '({x})'

    _doc_frontend_paragraph = \
        """
        This class inherits from `tf.keras.layers.Layer`. As a result, it has
        all the same capabilities as a standard Keras `Layer`.
        """

    _doc_sample = 'np.random.randn({shape})'

    _doc_has_shape = False

    _doc_has_out_type = False
