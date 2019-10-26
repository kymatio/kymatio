__all__ = ['Scattering1D']

import logging

class Scattering1D(object):
    def __init__(self, *args, **kwargs):
        if 'frontend' not in kwargs:
            frontend='numpy'
        else:
            frontend=kwargs['frontend'].lower()
            kwargs.pop('frontend')

        try:
            module = __import__(frontend + '_frontend', globals(), locals(), [], 1)
            if frontend == 'numpy':
                frontend = 'NumPy'
            elif frontend == 'tensorflow':
                frontend = 'TensorFlow'
            else:
                frontend = frontend.capitalize()
            self.__class__ = getattr(module, self.__class__.__name__ + frontend)
            self.__init__(*args, **kwargs)
        except Exception as e:
            raise e from RuntimeError('\nThe frontend \'' + frontend + '\' could not be correctly imported.')

        logging.info(self.loginfo())

