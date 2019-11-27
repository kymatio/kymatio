import logging
import warnings
import importlib


class Scattering2D(object):
    def __init__(self, *args, **kwargs):
        if 'frontend' not in kwargs:
            warnings.warn("Torch frontend is currently the default, but NumPy will become the default in the next"
                          " version.", PendingDeprecationWarning)
            frontend = 'torch'
        else:
            frontend = kwargs['frontend']
            kwargs.pop('frontend')

        try:
            module = importlib.__import__(frontend + '_frontend', globals(), locals(), [], 1)

            # Create frontend-specific class name by inserting frontend name
            # after `Scattering`.
            class_name = self.__class__.__name__
            class_name = (class_name[:-2] + frontend.capitalize()
                          + class_name[-2:])

            self.__class__ = getattr(module, class_name)
            self.__init__(*args, **kwargs)
        except Exception as e:
            raise e from RuntimeError('\nThe frontend \'' + frontend + '\' could not be correctly imported.')

        logging.info('The 2D frontend ' + frontend + ' was imported.')


__all__ = ['Scattering2D']
