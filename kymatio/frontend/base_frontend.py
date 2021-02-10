import importlib


class ScatteringBase():
    def __init__(self):
        super(ScatteringBase, self).__init__()

    def build(self):
        """ Defines elementary routines.

        This function should always call and create the filters via
        self.create_filters() defined below. For instance, via:
        self.filters = self.create_filters() """
        raise NotImplementedError

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
