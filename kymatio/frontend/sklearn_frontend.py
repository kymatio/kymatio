from sklearn.base import BaseEstimator, TransformerMixin


class ScatteringTransformerMixin(BaseEstimator, TransformerMixin):
    def fit(self, x=None, y=None):
        # No fitting necessary.
        return self

    def predict(self, x):
        n_samples = x.shape[0]
        x = x.reshape((-1,) + self.shape)
        Sx = self.scattering(x)
        Sx = Sx.reshape(n_samples, -1)

        return Sx

    transform = predict

    _doc_array = 'np.ndarray'
    _doc_array_n = 'n'

    _doc_alias_name = 'predict'

    _doc_alias_call = '.predict'

    _doc_frontend_paragraph = \
        """
        This class inherits from `BaseEstimator` and `TransformerMixin` in
        `sklearn.base`. As a result, it supports calculating the scattering
        transform by calling the `predict` and `transform` methods. By
        extension, it can be included as part of a scikit-learn `Pipeline`.
        """

    _doc_sample = 'np.random.randn(np.prod({shape}))'

    _doc_has_shape = True

    _doc_has_out_type = False
