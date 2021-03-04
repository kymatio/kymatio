import pytest
from kymatio import Scattering1D
from kymatio.scattering1d.filter_bank import scattering_filter_factory
import os
import numpy as np
import io

backends = []

from kymatio.scattering1d.backend.numpy_backend import backend
backends.append(backend)

class PlotFiltersT:    
    def Scattering1D_filter_factory_T_plot(self, N, J, T, phi_f, psi1_f, msg):
          import matplotlib.pyplot as plt
          plot_dir = 'plots'
          if not os.path.isdir(plot_dir):
              os.mkdir(plot_dir)    

          plt.figure()
          plt.plot(np.arange(N)/N, phi_f[0], 'r', label=msg)
          for psi_f in psi1_f:
              plt.plot(np.arange(N)/N, psi_f[0], 'b')
          plt.xlim(0, 0.5)
          plt.xlabel(r'$\omega$', fontsize=18)
          plt.ylabel(r'$\hat\psi_j(\omega)$', fontsize=18)
          
          plt.legend()
          plt.savefig(plot_dir + '/order1filters_T' + str(T) + 'J' + str(J) +  '.pdf', bbox_inches='tight', orientation='landscape')
    
    def Scattering1D_filter_factory_T(self, backend):
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
                msg = 'J=' + str(J) + ', T=' +str(T) + default_str + ': LP-filter width $\sigma_{low}$=' + str(phi_f['sigma']) + ' (=0.1/T)'                
                print(msg)
                assert(phi_f['sigma']==0.1/T)

                self.Scattering1D_filter_factory_T_plot(N, J, T, phi_f, psi1_f, msg)
                  
if __name__ == '__main__':
  test = PlotFiltersT()
  test.Scattering1D_filter_factory_T(backend)
    