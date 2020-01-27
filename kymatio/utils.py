from sklearn.base import BaseEstimator, TransformerMixin

import torch


class ScatteringTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, S, signal_shape, frontend='numpy'):
        """Creates an object that is compatible with the scikit-learn API
        and implements the .transform method.
        
        Parameters
        ==========

        S: an instance of Scattering1D, Scattering2D or Scattering3D
            This instance is called by the transformer

        signal_shape: tuple of ints
            The shape of one sample. Scikit-learn convention is to work with
            2D matrices only, of shape (n_samples, n_features). Data is
            delivered in this way and has to be reshaped before transform

        Output
        ======
        Y: ndarray, shape (n_samples, n_scattering_features)
            The scattering coefficients, raveled in C order
        """

        self.scattering = S
        self.signal_shape = signal_shape
        self.frontend = frontend

        assert frontend in ['numpy','torch','tensorflow']

    def fit(self, X=None, y=None):
        # no fitting necessary
        return self

    def predict(self, X):
        X_reshaped = X.reshape((-1,) + self.signal_shape)
        if self.frontend is 'torch':
            X_reshaped = torch.from_numpy(X_reshaped)
            if hasattr(self.scattering, "device"):
                X_reshaped = X_reshaped.to(self.scattering.device)

        transformed = self.scattering(X_reshaped)

        if self.frontend is 'torch':
            output = transformed.detach().cpu().numpy().reshape(X.shape[0], -1)
        else:
            output = transformed.reshape(X.shape[0], -1)

        return output

    transform = predict


