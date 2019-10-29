
class ScatteringSuper():
    def __init__(self):
        super(ScatteringSuper, self).__init__()
    
    def build(self):
        """ Defines elementary routines. 
        This function should always call and create the filters via
        self.create_filters() defined below. For instance, via:
        self.filters = self.create_filters() """
        raise NotImplementedError
    
    def create_filters(self):
        """ This function should run a filterbank function that
        will create the filters as numpy array, and then, it should
        save those arrays as module's buffers. """
        raise NotImplementedError

    def scattering(self, x):
        """ This function should call the functional scattering."""
        raise NotImplementedError
    
    def loginfo(self):
        """ Returns the logging message when the frontend is deployed."""
        raise NotImplementedError

