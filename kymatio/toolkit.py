# -*- coding: utf-8 -*-
"""Convenience utilities."""
import numpy as np


def pack_jtfs(out, jmeta, concat=False):
    """Pack coefficients into:
        - zo: zeroth-order
        - fo: first-order
        - phi_t * phi_f
        - psi_t * psi_f
        - psi_t * phi_f
        - phi_t * psi_f

    Currently only works with list `out`.
    """
    fo_idx0 = 1
    joint_idx0 = [i for i, n in enumerate(jmeta['n'])
                  if not np.any(np.isnan(n))][0]
    spin_down_idx0 = [i for i, s in enumerate(jmeta['s']) if s == -1][0]
    phi_f_idx0 = [i for i, s in enumerate(jmeta['s']) if s == 0][0]
    phi_t_idx0 = [i for i, n in enumerate(jmeta['n']) if n[0] == -1][0]

    packed = {}
    packed['zo'] = [out[0]['coef']]
    packed['fo'] = [out[i]['coef'] for i in range(fo_idx0, joint_idx0 - 1)]

    packed['phi_t * phi_f'] = [out[joint_idx0 - 1]['coef']]
    packed['psi_t * psi_f'] = [[], []]
    packed['psi_t * psi_f'][0] = [out[i]['coef'] for i in
                                  range(joint_idx0, spin_down_idx0)]
    packed['psi_t * psi_f'][1] = [out[i]['coef'] for i in
                                  range(spin_down_idx0, phi_f_idx0)]
    packed['psi_t * phi_f'] = [out[i]['coef'] for i in
                               range(phi_f_idx0, phi_t_idx0)]
    packed['phi_t * psi_f'] = [out[i]['coef'] for i in
                               range(phi_f_idx0, len(out))]

    if concat:
        for k, v in packed.items():
            if isinstance(v[0], list):
                for i in range(len(v)):
                    packed[k][i] = np.vstack(v[i])
            else:
                packed[k] = np.vstack(packed[k])
    return packed