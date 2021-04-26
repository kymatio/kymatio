import pytest
from tensorflow.keras.layers import Input, Flatten, Dense
from kymatio.keras import Scattering1D
from tensorflow.keras.models import Model
import os
import numpy as np
import io

backends = []

class TestScattering1DKeras:
    def test_Scattering1D(self):
        """
        Applies scattering on a stored signal to make sure its output agrees with
        a previously calculated version.
        """
        test_data_dir = os.path.dirname(__file__)

        with open(os.path.join(test_data_dir, 'test_data_1d.npz'), 'rb') as f:
            buffer = io.BytesIO(f.read())
            data = np.load(buffer)

        x = data['x']
        J = data['J']
        Q = data['Q']
        Sx0 = data['Sx']

        # default
        inputs1 = Input(shape=(x.shape[-1]))        
        scat1 = Scattering1D(J=J, Q=Q)(inputs1)
        
        model1 = Model(inputs1, scat1)        
        model1.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        Sg1 = model1.predict(x)
        assert np.allclose(Sg1, Sx0)
        
        # adjust T
        sigma_low_scale_factor = 2
        T=2**(J-sigma_low_scale_factor)

        inputs2 = Input(shape=(x.shape[-1]))        
        scat2 = Scattering1D(J=J, Q=Q, T=T)(inputs2)

        model2 = Model(inputs2, scat2)        
        model2.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        Sg2 = model2.predict(x)
        
        # for now, just be sure that the output shape is different from default
        # should we add this new result to the test data?
        #print('Sx1 shape: ' + str(Sx1.shape) + ' Sx2.shape: ' + str(Sx2.shape))
        assert Sg2.shape == (Sg1.shape[0], Sg1.shape[1], Sg1.shape[2]*2**(sigma_low_scale_factor))


if __name__ == '__main__':
  test = TestScattering1DKeras()
  test.test_Scattering1D_T()