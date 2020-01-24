import logging
import warnings
import importlib


class Scattering(object):
    def __init__(self, *args, **kwargs):
        self.name = kwargs['name']
        self.class_name = kwargs['class_name']
        kwargs.pop('name')
        kwargs.pop('class_name')

        frontend_suffixes = {'torch' : 'Torch', 'numpy' : 'NumPy', 'tensorflow'
                : 'TensorFlow'}
        if 'frontend' not in kwargs:
            warnings.warn("Torch frontend is currently the default, but NumPy will become the default in the next"
                          " version.", DeprecationWarning)
            frontend = 'torch'
        else:
            frontend = kwargs['frontend'].lower()
            kwargs.pop('frontend')

        try:
            module = importlib.import_module('kymatio.' + self.class_name + '.frontend.' + frontend + '_frontend')

            # Create frontend-specific class name by inserting frontend name
            # after `Scattering`.
            frontend = frontend_suffixes[frontend]

            class_name = self.__class__.__name__
            class_name = (class_name[:-2] + frontend + class_name[-2:])

            self.__class__ = getattr(module, class_name)
            self.__init__(*args, **kwargs)
        except Exception as e:
            raise e from RuntimeError('\nThe frontend \'' + frontend + '\' could not be correctly imported.')

        logging.info('The ' + self.name + ' frontend ' + frontend + ' was imported.')


__all__ = ['Scattering']
