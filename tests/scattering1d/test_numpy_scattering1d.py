import pytest
from kymatio import Scattering1D
from kymatio.scattering1d.filter_bank import scattering_filter_factory
import os
import numpy as np
import io

backends = []

from kymatio.scattering1d.backend.numpy_backend import backend
backends.append(backend)

class TestScattering1DNumpy:
    @pytest.mark.parametrize('backend', backends)
    def test_Scattering1D(self, backend):
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

        N = x.shape[-1]

        scattering = Scattering1D(J, N, Q, backend=backend, frontend='numpy')

        Sx = scattering(x)
        assert np.allclose(Sx, Sx0)
    
    def test_Scattering1D_T(self, backend):
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

        N = x.shape[-1]

        # default
        scattering1 = Scattering1D(J, N, Q, backend=backend, frontend='numpy')
        Sx1 = scattering1(x)
 
        # adjust T
        sigma_low_scale_factor = 2
        T=2**(J-sigma_low_scale_factor)
        scattering2 = Scattering1D(J, N, Q, T=T, backend=backend, frontend='numpy')
        Sx2 = scattering2(x)
        
        # for now, just be sure that the output shape is different from default
        # should we add this new result to the test data?
        #print('Sx1 shape: ' + str(Sx1.shape) + ' Sx2.shape: ' + str(Sx2.shape))
        assert Sx2.shape == (Sx1.shape[0], Sx1.shape[1], Sx1.shape[2]*2**(sigma_low_scale_factor))
        
    def test_Scattering1D_filter_factory_T(self, backend):
        """
        Constructs the scattering filters for the T parameter which controls the
        temporal extent of the low-pass sigma_log filter
        """
        N = 2**13
        Q = 1       
        sigma_low_scale_factor = [0, 5]        
        Js = [5]
        
        for j in Js:
            J = j
            for i in sigma_low_scale_factor:
                T=2**(J-i)
                if i == 0:
                    default_str = ' (default)'
                else: 
                    default_str = ''
                phi_f, psi1_f, psi2_f, _ = scattering_filter_factory(np.log2(N), J, Q, T)
                #msg = 'J=' + str(J) + ', T=' +str(T) + default_str + ': LP-filter width $\sigma_{low}$=' + str(phi_f['sigma']) + ' (=0.1/T)'                
                #print(msg)
                assert(phi_f['sigma']==0.1/T)

