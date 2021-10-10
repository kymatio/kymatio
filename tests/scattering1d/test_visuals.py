"""Test that visuals.py methods run without error."""
import pytest, os, warnings
from kymatio import Scattering1D, TimeFrequencyScattering1D
from kymatio.toolkit import echirp, pack_coeffs_jtfs
from kymatio import visuals
from utils import tempdir

# backend to use for most tests
default_backend = 'numpy'
# set True to execute all test functions without pytest
run_without_pytest = 0
# set True to disable matplotlib plots
# (done automatically for CI via `conftest.py`, but `False` here takes precedence)
no_plots = 1

# disable plots for pytest unit testing
# restart kernel when changing `no_plots`
if no_plots:
    import matplotlib
    matplotlib.use('template')
    matplotlib.pyplot.ioff()

# set up reusable references in global scope
sc_tms, jtfss, sc_all = [], [], []
metas = []
xs = []
out_tms, out_jtfss, out_all = [], [], []


def make_reusables():
    # run after __main__ so test doesn't fail during collection
    # reusable scattering objects
    N = 512
    kw0 = dict(shape=N, J=9, Q=8, frontend=default_backend)
    sc_tms.extend([Scattering1D(**kw0, out_type='array')])

    sfs = [('resample', 'resample'), ('exclude', 'resample'),
           ('recalibrate', 'recalibrate')]
    kw1 = dict(out_type='dict:array', **kw0)
    jtfss.extend([
        TimeFrequencyScattering1D(**kw1, sampling_filters_fr=sfs[0]),
        TimeFrequencyScattering1D(**kw1, sampling_filters_fr=sfs[1]),
        TimeFrequencyScattering1D(**kw1, sampling_filters_fr=sfs[2]),
    ])
    sc_all.extend([*sc_tms, *jtfss])

    # reusable input
    xs.append(echirp(N))
    # reusable outputs
    out_tms.extend([sc(xs[0]) for sc in sc_tms])
    out_jtfss.extend([jtfs(xs[0]) for jtfs in jtfss])
    out_all.extend([*out_tms, *out_jtfss])
    # metas
    metas.extend([sc.meta() for sc in sc_all])

    return sc_tms, jtfss, sc_all, metas, xs, out_tms, out_jtfss, out_all


#### Tests ###################################################################

def test_filterbank_heatmap(G):
    for i, sc in enumerate(sc_all):
        frequential = bool(i > 0)
        visuals.filterbank_heatmap(sc, first_order=True, second_order=True,
                                   frequential=frequential)


def test_filterbank_scattering(G):
    sc_all = G['sc_all']
    for sc in sc_all:
        visuals.filterbank_scattering(sc, second_order=1, lp_sum=1, zoom=0)
    for zoom in (4, -1):
        visuals.filterbank_scattering(sc_tms[0], second_order=1, lp_sum=1,
                                      zoom=zoom)



def test_filterbank_jtfs_1d(G):
    jtfss = G['jtfss']
    for jtfs in jtfss:
        visuals.filterbank_jtfs_1d(jtfs, lp_sum=1, zoom=0)
    for zoom in (4, -1):
        visuals.filterbank_jtfs_1d(jtfs, lp_sum=0, lp_phi=0, zoom=zoom)


def test_filterbank_jtfs_2d(G):
    jtfss = G['jtfss']
    visuals.filterbank_jtfs_2d(jtfss[1])


def test_gif_jtfs_2d(G):
    out_jtfss, metas = G['out_jtfss'], G['metas']
    savename = 'jtfs2d'
    fn = lambda savedir: visuals.gif_jtfs_2d(out_jtfss[1], metas[2], savedir=savedir,
                                          base_name=savename)
    _run_with_cleanup(fn, savename)


def test_gif_jtfs_3d(G):
    try:
        import plotly
    except ImportError:
        warnings.warn("Skipped `test_gif_jtfs_3d` since `plotly` not installed.")
        return

    out_jtfss, metas = G['out_jtfss'], G['metas']
    packed = pack_coeffs_jtfs(out_jtfss[1], metas[2], structure=2,
                              sampling_psi_fr='exclude')

    savename = 'jtfs3d'
    fn = lambda savedir: visuals.gif_jtfs_3d(packed, savedir=savedir,
                                             base_name=savename,
                                             images_ext='.png', verbose=False)
    _run_with_cleanup(fn, savename)


def test_energy_profile_jtfs(G):
    out_jtfss = G['out_jtfss']
    for i, Scx in enumerate(out_jtfss):
      for flatten in (False, True):
        for pairs in (None, ('phi_t * psi_f',)):
          test_params = dict(flatten=flatten, pairs=pairs)
          try:
              _ = visuals.energy_profile_jtfs(Scx, metas[1 + i], **test_params)
          except Exception as e:
              test_params['i'] = i
              print('\n'.join(f'{k}={v}' for k, v in test_params.items()))
              raise e


def test_coeff_distance_jtfs(G):
    out_jtfss = G['out_jtfss']
    for i, Scx in enumerate(out_jtfss):
      for flatten in (False, True):
        for pairs in (None, ('phi_t * psi_f',)):
          test_params = dict(flatten=flatten, pairs=pairs)
          try:
              _ = visuals.coeff_distance_jtfs(Scx, Scx, metas[1 + i],
                                              **test_params)
          except Exception as e:
              test_params['i'] = i
              print('\n'.join(f'{k}={v}' for k, v in test_params.items()))
              raise e


def _run_with_cleanup(fn, savename):
    with tempdir() as savedir:
        try:
            fn(savedir)
            path = os.path.join(savedir, savename + '.gif')
            # assert gif was made
            assert os.path.isfile(path), path
            os.unlink(path)
        finally:
            # clean up images, if any were made
            paths = [os.path.join(savedir, n) for n in os.listdir(savedir)
                     if (n.startswith(savename) and n.endswith('.png'))]
            for p in paths:
                os.unlink(p)



# create testing objects #####################################################
if run_without_pytest:
    sc_tms, jtfss, sc_all, metas, xs, out_tms, out_jtfss, out_all = (
        make_reusables())
    G = dict(sc_tms=sc_tms, jtfss=jtfss, sc_all=sc_all, metas=metas,
             xs=xs, out_tms=out_tms, out_jtfss=out_jtfss, out_all=out_all)
else:
    mr = [False]
    @pytest.fixture(scope='module')
    def G():
        if not mr[0]:
            sc_tms, jtfss, sc_all, metas, xs, out_tms, out_jtfss, out_all = (
                make_reusables())
            mr[0] = True
        return dict(sc_tms=sc_tms, jtfss=jtfss, sc_all=sc_all, metas=metas,
                    xs=xs, out_tms=out_tms, out_jtfss=out_jtfss, out_all=out_all)


# run tests ##################################################################
if __name__ == '__main__':
    if run_without_pytest:
        test_filterbank_heatmap(G)
        test_filterbank_scattering(G)
        test_filterbank_jtfs_1d(G)
        test_filterbank_jtfs_2d(G)
        test_gif_jtfs_2d(G)
        test_gif_jtfs_3d(G)
        test_energy_profile_jtfs(G)
        test_coeff_distance_jtfs(G)
    else:
        pytest.main([__file__, "-s"])
