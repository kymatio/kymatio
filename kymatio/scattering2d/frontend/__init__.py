__all__ = ['Scattering2D']

import logging

class Scattering2D(object):
    def __init__(self, *args, **kwargs):
        if 'frontend' not in kwargs:
            frontend='numpy'
        else:
            frontend=kwargs['frontend']
            kwargs.pop('frontend')

        try:
            module = __import__(frontend + '_frontend', globals(), locals(), [], 1)
            self.__class__ = getattr(module, self.__class__.__name__ + frontend.capitalize())
            self.__init__(*args, **kwargs)
        except Exception as e:
            raise e from RuntimeError('\nThe frontend \'' + frontend + '\' could not be correctly imported.')

        logging.info(self.loginfo())
