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
    
    def set_backend(self, string):
        """ This function should set the backend to be used if not already
        specified"""
        if isinstance(self.backend, str):
            self.backend = importlib.import_module(string + self.backend \
                    + "_backend", 'backend')
        elif not self.backend.name.startswith(self.backend_name):
            raise RuntimeError('This backend is not supported.')
 

    def create_filters(self):
        """ This function should run a filterbank function that
        will create the filters as numpy array, and then, it should
        save those arrays. """
        raise NotImplementedError

