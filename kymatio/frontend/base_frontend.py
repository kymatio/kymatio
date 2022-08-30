import importlib
import itertools


class ScatteringBase():
    def __init__(self):
        super(ScatteringBase, self).__init__()

    def build(self):
        """ Defines elementary routines.

        This function should always call and create the filters via
        self.create_filters() defined below. For instance, via:
        self.filters = self.create_filters() """
        raise NotImplementedError

    def _check_filterbanks(psi1s, psi2s):
        # Check for absence of aliasing at the first order
        # that the Nyquist frequency of each filter, 2**(-j), is at least
        # one octave above its frequency xi
        assert all((psi1["xi"] < 0.5/(2**psi1["j"])) for psi1 in psi1s)

        # Check for absence of aliasing at the second order
        # The filter function selects the (psi1, psi2) pairs such that j1<j2.
        # The map function evaluates whether those pairs also satisfy xi1>xi2.
        # Hence, the "assert all" statement guarantees that (j1<j2) => (xi1>xi2)
        psi_generator = itertools.product(psi1s, psi2s)
        condition = lambda psi1_or_2: psi1_or_2[0]['j'] < psi1_or_2[1]['j']
        implication = lambda psi1_or_2: psi1_or_2[0]['xi'] > psi1_or_2[1]['xi']
        assert all(map(implication, filter(condition, psi_generator)))

    def _instantiate_backend(self, import_string):
        """ This function should instantiate the backend to be used if not already
        specified"""

        # Either the user entered a string, in which case we load the corresponding backend.
        if isinstance(self.backend, str):
            if self.backend.startswith(self.frontend_name):
                try:
                    self.backend = importlib.import_module(import_string + self.backend + "_backend", 'backend').backend
                except ImportError:
                    raise ImportError('Backend ' + self.backend + ' not found!')
            else:
                raise ImportError('The backend ' + self.backend + ' can not be called from the frontend ' +
                                   self.frontend_name + '.')
        # Either the user passed a backend object, in which case we perform a compatibility check.
        else:
            if not self.backend.name.startswith(self.frontend_name):
                raise ImportError('The backend ' + self.backend.name + ' is not supported by the frontend ' +
                                   self.frontend_name + '.')

    def create_filters(self):
        """ This function should run a filterbank function that
        will create the filters as numpy array, and then, it should
        save those arrays. """
        raise NotImplementedError

    class _DryBackend:
        __getattr__ = lambda self, attr: (lambda *args, **kwargs: None)
