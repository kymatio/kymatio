import math
from ..backend.agnostic_backend import unpad_dyadic


def timefrequency_scattering1d(
        x, pad, unpad, backend, J, log2_T, psi1, psi2, phi, scf,
        pad_fn, pad_left=0, pad_right=0, ind_start=None, ind_end=None,
        oversampling=0, oversampling_fr=0, aligned=True, average=True,
        average_global=None, average_global_phi=None, out_type='array',
        out_3D=False, out_exclude=None, pad_mode='zero'):
    """
    Main function implementing the Joint Time-Frequency Scattering transform.

    Below is implementation documentation for developers.

    Frequential scattering
    ======================

    Variables
    ---------
    Explanation of variable naming and roles. For `scf`'s attributes see
    full Attributes docs in `TimeFrequencyScattering1D`.

    subsample_equiv_due_to_pad
        Equivalent amount of subsampling due to padding, relative to
        `J_pad_frs_max_init`.
        Distinguishes differences in input lengths (to `_frequency_scattering()`
        and `_joint_lowpass()`) due to padding and conv subsampling.
        This is needed to tell whether input is *trimmed* (pad difference)
        or *subsampled*.

    "conv subsampling"
        Refers to `subsample_fourier()` after convolution (as opposed to
        equivalent subsampling due to padding). i.e. "actual" subsampling.
        Refers to `total_conv_stride_over_U1`.

    total_conv_stride_over_U1
        See above. Alt name: `total_conv_stride_fr`; `over_U1` seeks
        to emphasize that it's the absolute stride over first order coefficients
        that matters rather than equivalent/relative quantities w.r.t. padded etc

    n1_fr_subsample
        Amount of conv subsampling done in `_frequency_scattering()`.

    total_downsample_fr
        Total amount of subsampling, conv and equivalent, relative to
        `J_pad_frs_max_init`. Conceptual quantity.

    Subsampling, padding
    --------------------
    Controlled by `aligned`, `out_3D`, `average_fr`, `log2_F`, and
    `sampling_psi_fr` & `sampling_phi_fr`.

    - `freq` == number of frequential rows (originating from U1), e.g. as in
      `(n1_fr, freq, time)` for joint slice shapes per `n2` (with `out_3D=True`).
    - `n1_fr` == number of frequential joint slices, or `psi1_f_fr_*`, per `n2`
    - `n2` == number of `psi2` wavelets, together with `n1_fr` controlling
      total number of joint slices

    aligned=True:
        Imposes:
          - `total_conv_stride_over_U1` to be same for all joint coefficients.
            Otherwise, row-to-row log-frequency differences, `dw,2*dw,...`,
            will vary across joint slices, which breaks alignment.
          - `sampling_psi_fr=='resample'`: center frequencies must be same
          - `sampling_phi_fr`: not necessarily restricted, as alignment is
            preserved under different amounts of frequential smoothing, but
            bins may blur together (rather unblur since `False` is finer);
            for same fineness/coarseness across slices, `True` is required.

        average_fr=False:
            - Additionally imposes that `total_conv_stride_over_U1==0`,
              since `psi_fr[0]['j'][0] == 0` and `_joint_lowpass()` can no longer
              compensate for variable `j_fr`.
            - Enforces separate `total_conv_stride_over_U1` for frequential
              lowpass coeffs: `phi_t * phi_f`, `psi_t * phi_f` will have their
              own common stride. This avoids significant redundancy while still
              complementing joint coeffs with lowpass information.

        out_3D=True:
            Additionally imposes that all frequential padding is the same
            (maximum), since otherwise same `total_conv_stride_over_U1`
            yields variable `freq` across `n2`.

        average_fr_global=True:
            Stride logic in `_joint_lowpass()` is relaxed since all U1 are
            collapsed into one point; will no longer check
            `total_conv_stride_over_U1`

    out_3D=True:
        Imposes:
            - `freq` to be same for all `n2` (can differ due to padding
              or convolutional stride)

    log2_F:
        Larger -> smaller `freq`
        Larger -> greater `max_subsample_before_phi_fr`
        Larger -> greater `J_pad_frs_max` (only if `log2_F > J_fr`)

    Debug tips
    ----------
      - Check following scf attributes:
          - N_frs
          - J_pad_frs, J_pad_frs_max_init
          - ind_end_fr, ind_end_fr_max
    """
    """
    In regular scattering, `log2_T` controls maximum total subsampling upon
    padded input; if padding beyond nextpow2, will unpad more, not subsample more

    Any reason for `total_conv_stride_over_U1` to not equal `log2_F`?
      - in `aligned=True` && `out_3D=False`, we pad variably, so min padded
        cannot stride same as max padded else we may collapse `freq` to < 0.
        Then what decides maximum stride?

        Note, if `log2_F` is 1 less than global averaging we get `freq=2`
        for max padded case, then this stride on min padded case would yield < 0.
        Thus we restrict maximum stride based on min padded case, and total
        strides (psi + phi) for all `n1_fr` must equal
        `log2_F - subsample_equiv_due_to_pad_min`.

      - in `aligned=True` && `out_3D=True` we always pad to maximum, thus
        we can (and should) have `total_conv_stride_over_U1 == log2_F`.

    Since we must have same `total_conv_stride_over_U1` for all `n1_fr`,
    in variably padded case we'll necessarily get variable `freq` per `n1_fr`,
    forbidding 3D concatenation (and which is why `out_3D=True` always pads same).

    In `aligned=False` we don't care about enforcing the same stride, so we
    subsample maximally for each padding - that is, as much as the filters will
    permit. The only restriction is, still, to not end up with <= 0 `freq`.
      - "Pad more -> unpad mode" still holds in fr scattering, so
        `total_downsample_fr_max` is set relative to `nextpow2(N_frs_max)`.
        That is, shorter `freq` due to the combination of padding less, and
        conv stride (subsampling), is the same for all paddings, and for greater
        `J_pad_fr` we simply unpad more.

    What is `log2_F` set relative to, and can `total_downsample_fr_max == log2_F`?
      - `log2_F` is set by the user indirectly via `F` to control amount of
        imposed frequency transposition invariance, via temporal support of
        lowpass.
      - `log2_F` controls max subsampling we can do after conv with `phi_fr`.

        What does this mean for inputs (to `phi_fr`) of varying length - namely,
        length lesser than maximum (which is the length `phi_fr` was originally
        sampled at, and length at which it could be subsampled by `2**log2_F`)?
          - With `resample_=True`, we can still subsample by `log2_F` without
            alias - restrictions will come from elsewhere
            (`aligned`, `freq` <= 0, etc).
          - With `resample_=False`, allowed subsampling is less than `log2_F`,
            and this is accounted for alongside existing constraints.

        Thus: account for `phi_fr`'s `j` at any length along other constraints.
      - `log2_F` cannot exceed `nextpow2(N_frs_max)`; can
        `total_downsample_fr_max` exceed it? For max padded case, whose pad length
        might exceed `nextpow2(N_frs_max)`, we can still total subsample by
        `log2_F` at most (`average_fr=True`; note this quantity is unused for
        `False`); the rest will "unpad more".

        Thus, `total_downsample_fr_max <= log2_F`. Can it be <? The *actual*
        subsampling we do can be, but it'll be due to factors unrelated to
        subsampling permitted by `phi_fr`.

        Thus, `total_downsample_fr_max == log2_F`. (effectively by definition)

      - However we still need a quantity that denotes *actual* maximum subsampling
        we'll do at any pad length, given configuration (`aligned` etc).
        Call it `total_downsample_fr_max_adj`.
          - Differs from `total_downsample_fr_max` when `aligned`: total conv
            stride is restricted by min padded case.
    """
    """
    Exhaustive diagrams
    ===================

    _joint_lowpass()
    ----------------

    **Conditions**:

    `phi_fr[subsample_equiv_due_to_pad][n1_fr_subsample]` indexing for all cases.
    What differs is subsampling afterwards, determined by:
        C1) `j` of the `phi_fr`
            - depends on `sampling_phi_fr`
        C2) whether `freq` should be same for all `n2, n1_fr`
            - depends on `out_3D`
        C3) whether `total_conv_stride_over_U1` should be same for all `n2, n1_fr`
            - depends on `aligned`
        C4) whether `freq` <= 0
            - independent of `aligned, out_3D, resample_*`
            - must enforce total subsampling (equivalent and due to stride)
              does not exceed `pad_fr`
            - indirect dependence on `sampling_phi_fr` where, in `True`,
              we can subsample by more than `log2_F`, which is accounted for
              by `max_subsample_before_phi_fr`

    **Definitions**:
        D1) `total_downsample_fr` refers to total subsampling (equivalent due to
            padding less, and due to convolutional stride) relative to
            `J_pad_frs_max_init`. That is,
            `total_downsample_fr == freq / J_pad_frs_max_init`, where `freq` is
            length of frequential dimension *before* unpadding. It is also ==
            `lowpass_subsample_fr + n1_fr_subsample + subsample_equiv_due_to_pad`.
        D2) `log2_N_fr == nextpow2(N_frs_max)` is the maximum possible
            `total_conv_stride_over_U1` for any configuration.
            (Exceeding would imply `freq` <0 0 over unpadded `N_frs_max`).
              - `log2_N_fr + diff == J_pad_frs_max_init` is the maximum possible
                `total_downsample_fr` for any configuration.
              - `log2_F <= log2_N_fr` for all configurations.
        D3) `sampling_psi_fr == True` shall refer to `== 'resample'`, and `False`
            to `'recalibrate'` or `'exclude'`. Likewise for phi (except there's no
            `'exclude'`).

    **Observations**:
        O1) With `aligned==True`, `lowpass_subsample_fr` must be a deterministic
            function of `n1_fr_subsample` (i.e. have same value at `n1_fr` for
            all `n2`), otherwise `total_conv_stride_over_U1` will vary over `n2`.
        O2) With `out_3D==True`, by similar logic, `total_downsample_fr` must
            be same for all `n1_fr, n2` to get same `freq`.

    ############################################################################
    X. aligned==True:
      __________________________________________________________________________
      A. out_3D==True:
        sampling_psi_fr, sampling_phi_fr:
        1. True, True:
           `lowpass_subsample_fr == log2_F - n1_fr_subsample`. Because:
            a. `lowpass_subsample_fr` cannot exceed `log2_F` because `log2_F`
                is the maximum subsampling factor of any `phi_fr`.
            b. `lowpass_subsample_fr` can be less than `log2_F` because must
               subsample less to get same `freq` with different `n1_fr_subsample`
            c. `lowpass_subsample_fr` and `n1_fr_subsample` must sum to the same
               (`total_conv_stride_over_U1`)
            d. `total_conv_stride_over_U1` cannot exceed `log2_F` because minimum
               `n1_fr_subsample == 0`, and max `lowpass_subsample_fr == log2_F`.
            e. `phi_fr` will be subsample-able by `log2_F` for any `pad_fr`
               (`J_pad_fr` will be restricted by `max_subsample_before_phi_fr`
                to ensure this).
            f. `freq` <= 0 isn't possible because we always have
               `pad_fr == J_pad_frs_max == J_pad_frs_max_init` (or equivalently
               `pad_fr == J_pad_frs_max <= J_pad_frs_max_init` if not
               `True, True`, where `J_pad_frs_max >= log2_N_fr`, and
               `log2_F <= log2_N_fr` (D2)).
            g. `max(..., 0)` isn't necessary since `n1_fr_subsample <= log2_F`
               is enforced (to avoid aliased `phi_fr`, note a;
               `subsample_equiv_due_to_pad + n1_fr_subsample` may exceed `log2_F`,
               but that won't require `phi_fr` subsampled beyond `log2_F` per
               `_, True`).
            h. `lowpass_subsample_fr`, `n1_fr_subsample`, and
               `subsample_equiv_due_to_pad` must sum to the same value
               (`total_downsample_fr`) for all `n2, n1_fr` (to get same `freq`).
               Since `subsample_equiv_due_to_pad` is same for all `n2, n1_fr`,
               only source of variability is `n1_fr_subsample`.

           `total_downsample_fr == log2_F`.
            i. `total_downsample_fr` cannot exceed `log2_F` due to
               `pad_fr == J_pad_frs_max == J_pad_frs_max_init`.

           `total_conv_stride_over_U1 == log2_F`
            d+. `lowpass_subsample_fr == log2_F` will happen along
                `n1_fr_subsample == 0`. Per c, `total_conv_stride_over_U1` hence
                also cannot be less, and must `== log2_F`.

        2. False, True:
           Same as 1. Because:
            a. Same as 1's. `J_pad_frs_max < J_pad_frs_max_init` is possible,
               but 1e applies.
            b,c,d,e,g,h: same as 1's.
            f. Same as 1's (`pad_fr == J_pad_frs_max >= log2_N_fr`).

           `total_downsample_fr == log2_F + diff`, where
           `diff == J_pad_frs_max_init - J_pad_frs_max`.
            i. `total_downsample_fr` can exceed `log2_F` because we can have
               `pad_fr == J_pad_frs_max < J_pad_frs_max_init`, and `_, True`
               will keep the `< 2**J_pad_frs_max_init` length `phi_fr`'s
               subsampling factor at `log2_F` (i.e. 1e).

           `total_conv_stride_over_U1 == log2_F`
            d+. Same as 1's (also see 2f).

        3. True, False:
           `lowpass_subsample_fr == (log2_F - diff) - n1_fr_subsample`. Because:
            a,b,c,h: same as 1's.
            d. same as 1's except `log2_F` -> `log2_F - diff` (see 3e).
            e. `phi_fr` will not be subsample-able by `log2_F` for all `pad_fr`s.
               `phi_fr` with `j=log2_F` is sampled at `2**J_pad_frs_max_init`,
               and subsequently has `j<log2_F` (see 3f).
               That is, `_, False` makes `phi_fr`'s `j` directly depend on input
               length, so both `n1_fr_subsample` and `subsample_equiv_due_to_pad`
               lower `j`. `_, False` (or `False, _`) enables
               `J_pad_frs_max < J_pad_frs_max_init`.
            f. same as 1's (+2's reasoning).
            g. same as 1's, except `n1_fr_subsample <= log2_F - diff` is enforced
               per c and 3d+.

            `total_downsample_fr == log2_F`. Because,
             i. Per 3e, exceeding `log2_F` would mean subsampling `phi_fr`
                beyond `log2_F`.

            `total_conv_stride_over_U1 == log2_F - diff`
             d+. Same as 1's, except `log2_F` -> `log2_F - diff` per 3e.

        4. False, False:
           Same as 3. `False, _` only affects `J_pad_fr`
           (but `pad_fr == J_pad_frs_max`, always) and `J_pad_frs_max`
           (and 3 accounts for this).

        ```
        lowpass_subsample_fr == phi_fr_sub_at_max - n1_fr_subsample
        phi_fr_sub_at_max == phi_fr['j'][J_pad_frs_max] <= log2_F
        # ^ maximum "realizable" subsampling after `phi_fr`
        ```
        accounts for 1-4.
          - `phi_fr_sub_at_max == log2_F` for 1,2 and we reduce to
            `lowpass_subsample_fr == log2_F - n1_fr_subsample`
          - `phi_fr_sub_at_max == log2_F - diff` for 3,4.

        ```
        total_downsample_fr == log2_F + (diff - j_diff)
        j_diff == phi_fr['j'][J_pad_frs_max_init] - phi_fr['j'][J_pad_frs_max]
        ```
        accounts for 1-4.
          - 1: `J_pad_frs_max == J_pad_frs_max_init` -> `diff == j_diff == 0`
               -> `total_downsample_fr == log2_F`
          - 2: `J_pad_frs_max <= J_pad_frs_max_init` -> `diff >= 0`, `j_diff == 0`
               -> `total_downsample_fr == log2_F + diff
          - 3: `J_pad_frs_max <= J_pad_frs_max_init` -> `diff == j_diff >= 0`
               -> `total_downsample_fr == log2_F`
          - 4: same as 3.

        ```
        total_conv_stride_over_U1 == log2_F - j_diff
                                  == phi_fr_sub_at_max
        ```
        accounts for 1-4.
          - `== log2_F` for 1,2, and `== log2_F - diff` for 3,4.
      __________________________________________________________________________
      B. out_3D==False:
         ^1: `pad_fr == J_pad_frs_max` no longer always holds
         ^2: same `freq` for all `n1_fr, n2` no longer required

        sampling_psi_fr, sampling_phi_fr:
        1. True, True
           `lowpass_subsample_fr == min(log2_F, J_pad_frs_min) - n1_fr_subsample`.
           Because:
            a. == A1a.
            b. `lowpass_subsample_fr` does not need to be less than `log2_F`
               due to B^2.
            c. == A1c. This overrides b.
            d. differs from A1d (in "max `lowpass_subsample_fr`"). See B1d+.
            e. == A1e.
            f. `freq` <= 0 is possible without `min` per B^1. `freq` <= 0 will
               occur upon `lowpass_subsample_fr > pad_fr`, so max `== pad_fr`.
               (namely `== J_pad_frs_min` per c).
            g. == A1g, except `n1_fr_subsample <= min(log2_F, J_pad_frs_min)`
               is enforced per c and B1d+ (or to avoid `lowpass_subsample_fr < 0`)
            h. N/A per B^2.

           `total_downsample_fr == (total_conv_stride_over_U1 +
                                    subsample_equiv_due_to_pad)`
           (this holds by definition; can plug expressions, but this is cleaner)
            i. `total_downsample_fr` can exceed `log2_F` because we can have
               `pad_fr < J_pad_frs_max == J_pad_frs_max_init` (or equivalently
               `pad_fr < J_pad_frs_max <= J_pad_frs_max_init` for
               not `True, True`).
               (`subsample_equiv_due_to_pad == J_pad_frs_max_init - pad_fr` is
               the generalization of `diff` (see e.g. A2g))

           `total_conv_stride_over_U1 == min(log2_F, J_pad_frs_min)`.
            d+. Smallest conv stride is determined from `J_pad_frs_min` such that
                `total_downsample_fr <= J_pad_frs_min`, i.e. sum of
                `lowpass_subsample_fr`, `n1_fr_subsample`, and
                `subsample_equiv_due_to_pad` is `<= J_pad_frs_min`.
                Smallest conv stride will occur at `n1_fr_subsample==0`, equal to
                `lowpass_subsample_fr`'s maximum at `J_pad_frs_min` (which is
                `log2_F` if `log2_F <= J_pad_frs_min`, else `J_pad_frs_min`).
                c then forbids exceeding this (and can't lower since this is min).

        2. False, True:
           Same as 1. `J_pad_frs_max < J_pad_frs_max_init` doesn't change any
           expression (but values within expressions might).

        3. True, False:
           `lowpass_subsample_fr == (log2_F - diffmin) - n1_fr_subsample`, where
           `diffmin == J_pad_frs_max_init - J_pad_frs_min`. Because:
            a,b,c,d,h: same as 1's.
            e. == A3e.
            f. `freq` <= 0 is prevented by `total_conv_stride_over_U1 ==`
               `log2_F - diffmin <= J_pad_frs_min`.
            g. same as A1's except `n1_fr_subsample <= log2_F - diffmin`
               is enforced per c and 3d+.

           `total_downsample_fr == (total_conv_stride_over_U1 +
                                    subsample_equiv_due_to_pad)`
            i. == A3i. Peaks when `pad_fr==J_pad_frs_min`, at `log2_F`.

           `total_conv_stride_over_U1 == log2_F - diffmin`.
            d+. == 1d+, except `lowpass_subsample_fr`'s maximum at `J_pad_frs_min`
                is `log2_F - diffmin`, and we always have
                `log2_F - diff <= J_pad_frs_min`.
                (Since `log2_F` occurs at J_pad_frs_max_init`, if
                `log2_F == log2_N_fr` (where `log2_N_fr <= J_pad_frs_max_init`),
                then at `J_pad_frs_min` it's
                `log2_N_fr - diffmin <= J_pad_frs_max_init - diffmin`
                `== J_pad_frs_min`. If `log2_F < log2_N_fr`, then
                `log2_F - diffmin < J_pad_frs_min` (instead of <=).)

        4. False, False:
            Same as 3. `False, _` only affects `pad_fr` via `J_pad_fr`
            (and 3 accounts for this).

        ```
        lowpass_subsample_fr == (min(phi_fr_sub_at_min, J_pad_frs_min) -
                                 n1_fr_subsample)
        phi_fr_sub_at_min == phi_fr['j'][k_at_min] <= log2_F;
        k_at_min = J_pad_frs_max_init - J_pad_frs_min
        # ^ subsampling after `phi_fr` in min-padded case
        ```
        accounts for 1-4.
          - `phi_fr_sub_at_min == log2_F` for 1,2 and we reduce to
            `lowpass_subsample_fr == min(log2_F, J_pad_frs_min) - n1_fr_subsample`
          - `phi_fr_sub_at_min == log2_F - diffmin <= J_pad_frs_min` for 3,4.

        ```
        total_downsample_fr == (subsample_equiv_due_to_pad +
                                total_conv_stride_over_U1)
        ```
        True by definition, but expressions for involved variables will vary.

        ```
        total_conv_stride_over_U1 == phi_fr_sub_at_min
        ```
        accounts for 1-4. Follows d+'s logic.

    Observe that in all cases, A and B, `lowpass_subsample_fr` results from
    satisfying C1,C3,C4, where we consider `total_conv_stride_over_U1` in min
    padded case, and set that as maximum permissible `lowpass_subsample_fr`.

    ############################################################################
    Y. aligned==False:

    This relaxes c, d, B^1 in all of X:
        c. `lowpass_subsample_fr` and `n1_fr_subsample` no longer need to sum
           to the same value (`total_conv_stride_over_U1`)
        d. `total_conv_stride_over_U1` can exceed `phi_fr_sub_at_max` per c.

    and applies B^1:
        B^1. `pad_fr == J_pad_frs_max` for all `n1_fr, n2` is only needed
             with `aligned and out_3D`.

           `lowpass_subsample_fr == min(log2_F, pad_fr) - n1_fr_subsample`.
      __________________________________________________________________________
      A. out_3D==True:
        sampling_psi_fr, sampling_phi_fr:
        1. True, True:
           `lowpass_subsample_fr = total_conv_stride_over_U1 - n1_fr_subsample`.
           This holds by definition; see j. Rationale:
            a,b,e: same as X.A.1's.
            f. == XB1f, except max is now `pad_fr` (varies over `n2`) rather
               than `J_pad_frs_min` (fixed) per relaxing c.
            g. See j.
            h. == XA1h, except `subsample_equiv_due_to_pad` varies. Thus
              `total_conv_stride_over_U1` must compensate.

           `total_downsample_fr == (total_conv_stride_over_U1 +
                                    subsample_equiv_due_to_pad)
            i. == XB1i. See j.

           `total_conv_stride_over_U1 = (min(log2_F, J_pad_frs_max) -
                                         (J_pad_frs_max - pad_fr))`
            j. max `total_conv_stride_over_U1` is determined in `J_pad_frs_max`
               case; for each `pad_fr` we can still subsample by up to `log2_F`,
               so lesser `pad_fr` -> greater `total_downsample_fr`, yielding
               different `freq`. To compensate, adjust `total_conv_stride_over_U1`
               relative to max padded case, to subsample less w/ lesser `pad_fr`.
               Expanding `lowpass_subsample_fr`:
                 `lowpass_subsample_fr = (min(log2_F, J_pad_frs_max) -
                                          (J_pad_frs_max - pad_fr) -
                                          n1_fr_subsample)`
               Thus we enforce `n1_fr_subsample <= (min(log2_F, J_pad_frs_max) -
                                                    (J_pad_frs_max - pad_fr))`
               to avoid `lowpass_subsample_fr < 0`.

        2. False, True
           Same as 1. See XB2.

        3. True, False
           `lowpass_subsample_fr = (min(log2_F, J_pad_frs_max) -
                                    subsample_equiv_due_to_pad -
                                    n1_fr_subsample)`.
           This is simply 1 with `J_pad_frs_max_init - pad_fr`, and `min` since
           `J_pad_frs_max < log2_F` is possible. It then follows:

           `n1_fr_subsample <= (min(log2_F, J_pad_frs_max) -
                                subsample_equiv_due_to_pad)`

           `total_conv_stride_over_U1 = (min(log2_F, J_pad_frs_max) -
                                         subsample_equiv_due_to_pad)`

        4. False, False
           Same as 3.

        ```
        lowpass_subsample_fr == (min(phi_fr_sub_at_max, J_pad_frs_max) -
                                 (J_pad_frs_max - pad_fr) - n1_fr_subsample)
        phi_fr_sub_at_max == phi_fr['j'][J_pad_frs_max]
        ```
        accounts for 1-4.

        ```
        total_conv_stride_over_U1 = (min(phi_fr_sub_at_max, J_pad_frs_max) -
                                     (J_pad_frs_max - pad_fr))`
        total_downsample_fr = (total_conv_stride_over_U1 +
                               subsample_equiv_due_to_pad)
        ```

    Example w/ `True, True`:
        J_pad_frs_min = 7
        J_pad_frs_max = 10
        J_pad_frs_max_init = 11
        log2_F = 5
        j1_fr_max = 6

        pad_fr == J_pad_frs_min == 7:
            subsample_equiv_due_to_pad = 11 - 7 = 4
            J_pad_frs_max - pad_fr = 3

            j1_fr == 6:
                n1_fr_subsample = 5 - 3 = 2
                lowpass_subsample_fr = 5 - 3 - 2 = 0
                total_conv_stride_over_U1 = 2 + 0 = 2
                total_downsample_fr = 2 + 4 = 6

            j1_fr == 0:
                n1_fr_subsample = 0
                lowpass_subsample_fr = 5 - 3 - 2 = 2
                total_conv_stride_over_U1 = 0 + 2 = 2
                total_downsample_fr = 2 + 4 = 6

        pad_fr == J_pad_frs_max == 10:
            subsample_equiv_due_to_pad = 1
            j1_fr == 6:
                n1_fr_subsample = 5
                lowpass_subsample_fr = 0
                total_conv_stride_over_U1 = 5 + 0 = 5
                total_downsample_fr = 5 + 1 = 6

            j1_fr == 0:
                n1_fr_subsample = 0
                lowpass_subsample_fr = 5
                total_conv_stride_over_U1 = 0 + 5 = 5
                total_downsample_fr = 5 + 1 = 6

      __________________________________________________________________________
      B. out_3D==False:
         XB^2 relaxed.

         This leaves only one condition: "not `freq` <= 0". We can now subsample
         maximally at every stage:

         ```
         lowpass_subsample_fr = j_phi - n1_fr_subsample
         total_conv_stride_over_U1 = j_phi
         total_downsample_fr = j_phi + subsample_equiv_due_to_pad
         n1_fr_subsample <= j_phi
         ```
         accounts for 1-4.

    ############################################################################

    Accounting for all of X and Y:

    ```
    k = subsample_equiv_due_to_pad
    if aligned:
        if out_3D:
            total_conv_stride_over_U1 = phi_fr['j'][k]
        else:
            k_at_min = J_pad_frs_max_init - J_pad_frs_min
            total_conv_stride_over_U1 = min(phi_fr['j'][k_at_min],
                                            J_pad_frs_min)
    else:
        if out_3D:
            total_conv_stride_over_U1 = (min(phi_fr['j'][k], J_pad_frs_max) -
                                         (J_pad_frs_max - pad_fr))
        else:
            total_conv_stride_over_U1 = phi_fr['j'][k]

    n1_fr_subsample = min(j1_fr, total_conv_stride_over_U1)
    lowpass_subsample_fr = total_conv_stride_over_U1 - n1_fr_subsample
    total_downsample_fr = n1_fr_subsample + lowpass_subsample_fr + k
    ```
    """
    # pack for later
    B = backend
    average_fr = scf.average_fr
    if out_exclude is None:
        out_exclude = []
    N = x.shape[-1]
    commons = (B, scf, out_exclude, aligned, oversampling_fr, average_fr,
               out_3D, oversampling, average, average_global, average_global_phi,
               unpad, log2_T, phi, ind_start, ind_end, N)

    out_S_0 = []
    out_S_1_tm = []
    out_S_1 = {'phi_t * phi_f': []}
    out_S_2 = {'psi_t * psi_f': [[], []],
               'psi_t * phi_f': [],
               'phi_t * psi_f': [[]]}

    # pad to a dyadic size and make it complex
    U_0 = pad_fn(x)
    # compute the Fourier transform
    U_0_hat = B.rfft(U_0)

    # for later
    J_pad = math.log2(U_0.shape[-1])
    commons2 = average, log2_T, J, J_pad, N, ind_start, ind_end, unpad, phi

    # Zeroth order ###########################################################
    if 'S0' not in out_exclude:
        if average_global:
            k0 = log2_T
            S_0 = B.mean(U_0, axis=-1)
        elif average:
            k0 = max(log2_T - oversampling, 0)
            S_0_c = B.cdgmm(U_0_hat, phi[0][0])
            S_0_hat = B.subsample_fourier(S_0_c, 2**k0)
            S_0_r = B.irfft(S_0_hat)
            S_0 = unpad(S_0_r, ind_start[0][k0], ind_end[0][k0])
        else:
            S_0 = x
        if average:
            S_0 *= B.sqrt(2**k0, dtype=S_0.dtype)  # subsampling energy correction
        out_S_0.append({'coef': S_0,
                        'j': (log2_T,) if average else (),
                        'n': (-1,)     if average else (),
                        's': (),
                        'stride': (k0,)  if average else (),})

    # First order ############################################################
    def compute_U_1(k1):
        U_1_c = B.cdgmm(U_0_hat, psi1[n1][0])
        U_1_hat = B.subsample_fourier(U_1_c, 2**k1)
        U_1_c = B.ifft(U_1_hat)

        # Modulus
        U_1_m = B.modulus(U_1_c)

        # Map to Fourier domain
        U_1_hat = B.rfft(U_1_m)
        return U_1_hat, U_1_m

    include_phi_t = any(pair not in out_exclude for pair in
                        ('phi_t * phi_f', 'phi_t * psi_f'))
    U_1_hat_list, S_1_tm_list = [], []
    for n1 in range(len(psi1)):
        # Convolution + subsampling
        j1 = psi1[n1]['j']
        sub1_adj = min(j1, log2_T) if average else j1
        k1 = max(sub1_adj - oversampling, 0)

        U_1_hat, U_1_m = compute_U_1(k1)
        U_1_hat_list.append(U_1_hat)

        # if `k1` is used from this point, treat as if `average=True`
        sub1_adj_avg = min(j1, log2_T)
        k1_avg = max(sub1_adj_avg - oversampling, 0)
        if average or include_phi_t:
            k1_J = (max(log2_T - k1_avg - oversampling, 0)
                    if not average_global_phi else log2_T - k1_avg)
            ind_start_tm_avg = ind_start[0][k1_J + k1_avg]
            ind_end_tm_avg   = ind_end[  0][k1_J + k1_avg]
            if not average_global_phi:
                if k1 != k1_avg:
                    # must recompute U_1_hat
                    U_1_hat_avg, _ = compute_U_1(k1_avg)
                else:
                    U_1_hat_avg = U_1_hat
                # Low-pass filtering over time
                S_1_c = B.cdgmm(U_1_hat_avg, phi[0][k1_avg])

                S_1_hat = B.subsample_fourier(S_1_c, 2**k1_J)
                S_1_avg = B.irfft(S_1_hat)
                # unpad since we're fully done with convolving over time
                S_1_avg = unpad(S_1_avg, ind_start_tm_avg, ind_end_tm_avg)
                total_conv_stride_tm_avg = k1_avg + k1_J
            else:
                # Average directly
                S_1_avg = B.mean(U_1_m, axis=-1)
                total_conv_stride_tm_avg = log2_T

        if 'S1' not in out_exclude:
            if average:
                ind_start_tm, ind_end_tm = ind_start_tm_avg, ind_end_tm_avg
            if average_global:
                S_1_tm = S_1_avg
            elif average:
                # Unpad averaged
                S_1_tm = S_1_avg
            else:
                # Unpad unaveraged
                ind_start_tm, ind_end_tm = ind_start[0][k1], ind_end[0][k1]
                S_1_tm = unpad(U_1_m, ind_start_tm, ind_end_tm)
            total_conv_stride_tm = (total_conv_stride_tm_avg if average else
                                    k1)

            # energy correction due to stride & inexact unpad length
            S_1_tm = _energy_correction(S_1_tm, B,
                                        param_tm=(N, ind_start_tm, ind_end_tm,
                                                  total_conv_stride_tm))
            out_S_1_tm.append({'coef': S_1_tm, 'j': (j1,), 'n': (n1,), 's': (),
                               'stride': (total_conv_stride_tm,)})

            # since tensorflow won't update it in `_energy_correction`
            if average:
                S_1_avg = S_1_tm

        # append for further processing
        if include_phi_t:
            # energy correction, if not done
            S_1_avg_energy_corrected = bool(average and 'S1' not in out_exclude)
            if not S_1_avg_energy_corrected:
                S_1_avg = _energy_correction(
                    S_1_avg, B, param_tm=(N, ind_start_tm_avg, ind_end_tm_avg,
                                          total_conv_stride_tm_avg))
            S_1_tm_list.append(S_1_avg)

    # Frequential averaging over time averaged coefficients ##################
    # `U1 * (phi_t * phi_f)` pair
    if include_phi_t:
        # zero-pad along frequency
        pad_fr = scf.J_pad_frs_max
        S_1_tm = _right_pad(S_1_tm_list, pad_fr, scf, B)

        if (('phi_t * phi_f' not in out_exclude and
             not scf.average_fr_global_phi) or
                'phi_t * psi_f' not in out_exclude):
            # map frequency axis to Fourier domain
            S_1_tm_hat = B.rfft(S_1_tm, axis=-2)

    if 'phi_t * phi_f' not in out_exclude:
        n1_fr_subsample = 0  # no intermediate scattering

        if scf.average_fr_global_phi:
            # take mean along frequency directly
            S_1 = B.mean(S_1_tm, axis=-2)
            lowpass_subsample_fr = scf.log2_F
        else:
            # this is usually 0
            subsample_equiv_due_to_pad = scf.J_pad_frs_max_init - pad_fr

            j1_fr = scf.phi_f_fr['j'][subsample_equiv_due_to_pad]
            # ensure stride is zero if `not average and aligned`
            average_fr = bool(scf.average_fr or not aligned)
            total_conv_stride_over_U1 = _get_stride(
                j1_fr, pad_fr, subsample_equiv_due_to_pad, scf, average_fr)
            lowpass_subsample_fr = max(total_conv_stride_over_U1 -
                                       n1_fr_subsample - oversampling_fr, 0)

            # Low-pass filtering over frequency
            phi_fr = scf.phi_f_fr[subsample_equiv_due_to_pad][n1_fr_subsample]
            S_1_c = B.cdgmm(S_1_tm_hat, phi_fr)
            S_1_hat = B.subsample_fourier(S_1_c, 2**lowpass_subsample_fr,
                                          axis=-2)
            S_1_c = B.irfft(S_1_hat, axis=-2)

        # compute for unpad / energy correction
        _stride = n1_fr_subsample + lowpass_subsample_fr
        if out_3D:
            ind_start_fr = scf.ind_start_fr_max[_stride]
            ind_end_fr   = scf.ind_end_fr_max[  _stride]
        else:
            ind_start_fr = scf.ind_start_fr[-1][_stride]
            ind_end_fr   = scf.ind_end_fr[-1][  _stride]

        # Unpad frequency
        if not scf.average_fr_global_phi:
            S_1 = unpad(S_1_c, ind_start_fr, ind_end_fr, axis=-2)

        # set reference for later
        total_conv_stride_over_U1_realized = (n1_fr_subsample +
                                              lowpass_subsample_fr)

        # energy correction due to stride & inexact unpad indices
        # time already done
        S_1 = _energy_correction(S_1, B,
                                 param_fr=(scf.N_frs_max,
                                           ind_start_fr, ind_end_fr,
                                           total_conv_stride_over_U1_realized))

        scf.__total_conv_stride_over_U1 = total_conv_stride_over_U1_realized
        # append to out with meta
        j1_fr = (scf.log2_F if scf.average_fr_global_phi else
                 scf.phi_f_fr['j'][subsample_equiv_due_to_pad])
        stride = (total_conv_stride_over_U1_realized, log2_T)
        out_S_1['phi_t * phi_f'].append({
            'coef': S_1, 'j': (log2_T, j1_fr), 'n': (-1, -1), 's': (0,),
            'stride': stride})
    else:
        scf.__total_conv_stride_over_U1 = -1

    ##########################################################################
    # Joint scattering: separable convolutions (along time & freq), and low-pass
    # `U1 * (psi_t * psi_f)` (up & down), and `U1 * (psi_t * phi_f)`
    skip_spinned = bool('psi_t * psi_f_up'   in out_exclude and
                        'psi_t * psi_f_down' in out_exclude)
    if not (skip_spinned and 'psi_t * phi_f' in out_exclude):
        for n2 in range(len(psi2)):
            j2 = psi2[n2]['j']
            if j2 == 0:
                continue

            Y_2_list = []
            # Wavelet transform over time
            for n1 in range(len(psi1)):
                # Retrieve first-order coefficient in the list
                j1 = psi1[n1]['j']
                if j1 >= j2:
                    continue
                U_1_hat = U_1_hat_list[n1]

                # what we subsampled in 1st-order
                sub1_adj = min(j1, log2_T) if average else j1
                k1 = max(sub1_adj - oversampling, 0)
                # what we subsample now in 2nd
                sub2_adj = min(j2, log2_T) if average else j2
                k2 = max(sub2_adj - k1 - oversampling, 0)

                # Convolution and downsampling
                Y_2_c = B.cdgmm(U_1_hat, psi2[n2][k1])
                Y_2_hat = B.subsample_fourier(Y_2_c, 2**k2)
                Y_2_c = B.ifft(Y_2_hat)

                # sum is same for all `n1`
                k1_plus_k2 = k1 + k2
                Y_2_c, trim_tm = _maybe_unpad_time(Y_2_c, k1_plus_k2, commons2)
                Y_2_list.append(Y_2_c)

            # frequential pad
            if aligned and out_3D:
                pad_fr = scf.J_pad_frs_max
            else:
                pad_fr = scf.J_pad_frs[n2]

            if scf.pad_mode_fr == 'custom':
                Y_2_arr = scf.pad_fn_fr(Y_2_list, pad_fr, scf, B)
            else:
                Y_2_arr = _right_pad(Y_2_list, pad_fr, scf, B)

            # temporal pad modification
            if pad_mode == 'reflect' and average:
                # `=` since tensorflow makes copy
                Y_2_arr = B.conj_reflections(Y_2_arr,
                                             ind_start[trim_tm][k1_plus_k2],
                                             ind_end[  trim_tm][k1_plus_k2],
                                             k1_plus_k2, N,
                                             pad_left, pad_right, trim_tm)

            # swap axes & map to Fourier domain to prepare for conv along freq
            Y_2_hat = B.fft(Y_2_arr, axis=-2)

            # Transform over frequency + low-pass, for both spins ############
            # `* psi_f` part of `U1 * (psi_t * psi_f)`
            if not skip_spinned:
                _frequency_scattering(Y_2_hat, j2, n2, pad_fr, k1_plus_k2,
                                      trim_tm, commons, out_S_2['psi_t * psi_f'])

            # Low-pass over frequency ########################################
            # `* phi_f` part of `U1 * (psi_t * phi_f)`
            if 'psi_t * phi_f' not in out_exclude:
                _frequency_lowpass(Y_2_hat, Y_2_arr, j2, n2, pad_fr, k1_plus_k2,
                                   trim_tm, commons, out_S_2['psi_t * phi_f'])

    ##########################################################################
    # `U1 * (phi_t * psi_f)`
    if 'phi_t * psi_f' not in out_exclude:
        # take largest subsampling factor
        j2 = log2_T
        k1_plus_k2 = (max(log2_T - oversampling, 0) if not average_global_phi else
                      log2_T)
        pad_fr = scf.J_pad_frs_max
        # n2_time = U_0.shape[-1] // 2**max(j2 - oversampling, 0)

        # reuse from first-order scattering
        Y_2_hat = S_1_tm_hat

        # Transform over frequency + low-pass
        # `* psi_f` part of `U1 * (phi_t * psi_f)`
        _frequency_scattering(Y_2_hat, j2, -1, pad_fr, k1_plus_k2, 0, commons,
                              out_S_2['phi_t * psi_f'], spin_down=False)

    ##########################################################################
    # pack outputs & return
    out = {}
    out['S0'] = out_S_0
    out['S1'] = out_S_1_tm
    out['phi_t * phi_f'] = out_S_1['phi_t * phi_f']
    out['phi_t * psi_f'] = out_S_2['phi_t * psi_f'][0]
    out['psi_t * phi_f'] = out_S_2['psi_t * phi_f']
    out['psi_t * psi_f_up']   = out_S_2['psi_t * psi_f'][0]
    out['psi_t * psi_f_down'] = out_S_2['psi_t * psi_f'][1]

    # delete excluded
    for pair in out_exclude:
        del out[pair]

    # warn of any zero-sized coefficients
    for pair in out:
        for i, c in enumerate(out[pair]):
            if 0 in c['coef'].shape:
                import warnings
                warnings.warn("out[{}][{}].shape == {}".format(
                    pair, i, c['coef'].shape))

    # concat
    if out_type == 'dict:array':
        for k, v in out.items():
            if out_3D:
                # stack joint slices, preserve 3D structure
                out[k] = B.concatenate([c['coef'] for c in v], axis=1)
            else:
                # flatten joint slices, return 2D
                out[k] = B.concatenate_v2([c['coef'] for c in v], axis=1)
    elif out_type == 'dict:list':
        pass  # already done
    elif out_type == 'array':
        if out_3D:
            # cannot concatenate `S0` & `S1` with joint slices, return two arrays
            out_0 = B.concatenate_v2([c['coef'] for k, v in out.items()
                                      for c in v if k in ('S0', 'S1')], axis=1)
            out_1 = B.concatenate([c['coef'] for k, v in out.items()
                                   for c in v if k not in ('S0', 'S1')], axis=1)
            out = (out_0, out_1)
        else:
            # flatten all and concat along `freq` dim
            out = B.concatenate_v2([c['coef'] for v in out.values()
                                    for c in v], axis=1)
    elif out_type == 'list':
        out = [c for v in out.values() for c in v]

    return out


def _frequency_scattering(Y_2_hat, j2, n2, pad_fr, k1_plus_k2, trim_tm, commons,
                          out_S_2, spin_down=True):
    (B, scf, out_exclude, aligned, oversampling_fr, average_fr, out_3D, *_
     ) = commons

    psi1_fs, spins = [], []
    if ('psi_t * psi_f_up' not in out_exclude or
            'psi_t * phi_f' not in out_exclude):
        psi1_fs.append(scf.psi1_f_fr_up)
        spins.append(1 if spin_down else 0)
    if spin_down and 'psi_t * psi_f_down' not in out_exclude:
        psi1_fs.append(scf.psi1_f_fr_down)
        spins.append(-1)

    subsample_equiv_due_to_pad = scf.J_pad_frs_max_init - pad_fr

    # Transform over frequency + low-pass, for both spins (if `spin_down`)
    for s1_fr, (spin, psi1_f) in enumerate(zip(spins, psi1_fs)):
        for n1_fr in range(len(psi1_f)):
            if scf.sampling_psi_fr == 'exclude':
                # 'exclude' is scale-oriented but to be safe also check
                # if padded length is available
                if subsample_equiv_due_to_pad not in psi1_f[n1_fr]:
                    continue
                width = psi1_f[n1_fr]['width'][subsample_equiv_due_to_pad]
                if width > scf.N_frs[n2]:
                    continue

            # compute subsampling
            j1_fr = psi1_f[n1_fr]['j'][subsample_equiv_due_to_pad]
            total_conv_stride_over_U1 = _get_stride(
                j1_fr, pad_fr, subsample_equiv_due_to_pad, scf, average_fr)

            if average_fr:
                sub_adj = min(j1_fr, total_conv_stride_over_U1,
                              scf.max_subsample_before_phi_fr[n2])
            else:
                sub_adj = min(j1_fr, total_conv_stride_over_U1)
            n1_fr_subsample = max(sub_adj - oversampling_fr, 0)

            # Wavelet transform over frequency
            Y_fr_c = B.cdgmm(Y_2_hat, psi1_f[n1_fr][subsample_equiv_due_to_pad])
            Y_fr_hat = B.subsample_fourier(Y_fr_c, 2**n1_fr_subsample, axis=-2)
            Y_fr_c = B.ifft(Y_fr_hat, axis=-2)

            # Modulus
            U_2_m = B.modulus(Y_fr_c)

            # Convolve by Phi = phi_t * phi_f, unpad
            S_2, stride = _joint_lowpass(
                U_2_m, n2, n1_fr, subsample_equiv_due_to_pad, n1_fr_subsample,
                k1_plus_k2, total_conv_stride_over_U1, trim_tm, commons)

            # append to out
            out_S_2[s1_fr].append(
                {'coef': S_2, 'j': (j2, j1_fr), 'n': (n2, n1_fr), 's': (spin,),
                 'stride': stride})


def _frequency_lowpass(Y_2_hat, Y_2_arr, j2, n2, pad_fr, k1_plus_k2, trim_tm,
                       commons, out_S_2):
    B, scf, _, aligned, oversampling_fr, average_fr, *_ = commons

    subsample_equiv_due_to_pad = scf.J_pad_frs_max_init - pad_fr

    if scf.average_fr_global_phi:
        Y_fr_c = B.mean(Y_2_arr, axis=-2)
        j1_fr = scf.log2_F
        # `min` in case `pad_fr > N_fr_scales_max`
        total_conv_stride_over_U1 = min(pad_fr, scf.log2_F)
        n1_fr_subsample = total_conv_stride_over_U1
    else:
        j1_fr = scf.phi_f_fr['j'][subsample_equiv_due_to_pad]
        # ensure stride is zero if `not average and aligned`
        average_fr = bool(scf.average_fr or not aligned)
        total_conv_stride_over_U1 = _get_stride(
            j1_fr, pad_fr, subsample_equiv_due_to_pad, scf, average_fr)
        if average_fr:
            sub_adj = min(j1_fr, total_conv_stride_over_U1,
                          scf.max_subsample_before_phi_fr[n2])
        else:
            sub_adj = min(j1_fr, total_conv_stride_over_U1)
        n1_fr_subsample = max(sub_adj - oversampling_fr, 0)

        Y_fr_c = B.cdgmm(Y_2_hat, scf.phi_f_fr[subsample_equiv_due_to_pad][0])
        Y_fr_hat = B.subsample_fourier(Y_fr_c, 2**n1_fr_subsample, axis=-2)
        Y_fr_c = B.ifft(Y_fr_hat, axis=-2)

    # Modulus
    U_2_m = B.modulus(Y_fr_c)

    # Convolve by Phi = phi_t * phi_f
    S_2, stride = _joint_lowpass(U_2_m, n2, -1, subsample_equiv_due_to_pad,
                                 n1_fr_subsample, k1_plus_k2,
                                 total_conv_stride_over_U1, trim_tm, commons)

    out_S_2.append({'coef': S_2, 'j': (j2, j1_fr), 'n': (n2, -1), 's': (0,),
                    'stride': stride})


def _joint_lowpass(U_2_m, n2, n1_fr, subsample_equiv_due_to_pad, n1_fr_subsample,
                   k1_plus_k2, total_conv_stride_over_U1, trim_tm, commons):
    (B, scf, _, aligned, oversampling_fr, average_fr, out_3D, oversampling,
     average, average_global, average_global_phi, unpad, log2_T, phi,
     ind_start, ind_end, N) = commons

    # compute subsampling logic ##############################################
    global_averaged_fr = (scf.average_fr_global if n1_fr != -1 else
                          scf.average_fr_global_phi)
    if global_averaged_fr:
        lowpass_subsample_fr = total_conv_stride_over_U1 - n1_fr_subsample
    elif average_fr:
        lowpass_subsample_fr = max(total_conv_stride_over_U1 - n1_fr_subsample -
                                   oversampling_fr, 0)
    else:
        lowpass_subsample_fr = 0

    # compute freq unpad params ##############################################
    do_averaging    = bool(average    and n2    != -1)
    do_averaging_fr = bool(average_fr and n1_fr != -1)
    total_conv_stride_over_U1_realized = n1_fr_subsample + lowpass_subsample_fr

    # freq params
    if out_3D:
        pad_ref = (scf.J_pad_frs_min if aligned else
                   scf.J_pad_frs_max)
        subsample_equiv_due_to_pad_ref = scf.J_pad_frs_max_init - pad_ref
        # ensure stride is zero if `not average and aligned`
        average_fr = bool(scf.average_fr or not aligned)
        stride_ref = _get_stride(
            None, pad_ref, subsample_equiv_due_to_pad_ref, scf, average_fr)
        stride_ref = max(stride_ref - oversampling_fr, 0)
        ind_start_fr = scf.ind_start_fr_max[stride_ref]
        ind_end_fr   = scf.ind_end_fr_max[  stride_ref]
    else:
        _stride = total_conv_stride_over_U1_realized
        ind_start_fr = scf.ind_start_fr[n2][_stride]
        ind_end_fr   = scf.ind_end_fr[  n2][_stride]

    # unpad early if possible ################################################
    # `not average` and `n2 == -1` already unpadded
    if not do_averaging:
        pass
    if not do_averaging_fr:
        U_2_m = unpad(U_2_m, ind_start_fr, ind_end_fr, axis=-2)
    unpadded_tm, unpadded_fr = (not do_averaging), (not do_averaging_fr)

    # freq lowpassing ########################################################
    if do_averaging_fr:
        if scf.average_fr_global:
            S_2_fr = B.mean(U_2_m, axis=-2)
        elif average_fr:
            # Low-pass filtering over frequency
            phi_fr = scf.phi_f_fr[subsample_equiv_due_to_pad][n1_fr_subsample]
            U_2_hat = B.rfft(U_2_m, axis=-2)
            S_2_fr_c = B.cdgmm(U_2_hat, phi_fr)
            S_2_fr_hat = B.subsample_fourier(S_2_fr_c, 2**lowpass_subsample_fr,
                                             axis=-2)
            S_2_fr = B.irfft(S_2_fr_hat, axis=-2)
    else:
        S_2_fr = U_2_m

    # unpad only if input isn't global averaged
    if not global_averaged_fr and not unpadded_fr:
        S_2_fr = unpad(S_2_fr, ind_start_fr, ind_end_fr, axis=-2)

    # time lowpassing ########################################################
    if do_averaging:
        if average_global:
            S_2_r = B.mean(S_2_fr, axis=-1)
            total_conv_stride_tm = log2_T
        elif average:
            # Low-pass filtering over time
            k2_tm_J = max(log2_T - k1_plus_k2 - oversampling, 0)
            U_2_hat = B.rfft(S_2_fr)
            S_2_c = B.cdgmm(U_2_hat, phi[trim_tm][k1_plus_k2])
            S_2_hat = B.subsample_fourier(S_2_c, 2**k2_tm_J)
            S_2_r = B.irfft(S_2_hat)
            total_conv_stride_tm = k1_plus_k2 + k2_tm_J
    else:
        total_conv_stride_tm = k1_plus_k2
        S_2_r = S_2_fr
    ind_start_tm = ind_start[trim_tm][total_conv_stride_tm]
    ind_end_tm   = ind_end[  trim_tm][total_conv_stride_tm]

    # `not average` and `n2 == -1` already unpadded
    if not average_global and not unpadded_tm:
        S_2_r = unpad(S_2_r, ind_start_tm, ind_end_tm)
    S_2 = S_2_r

    # energy correction ######################################################
    param_tm = (N, ind_start_tm, ind_end_tm, total_conv_stride_tm)
    param_fr = (scf.N_frs[n2], ind_start_fr,
                ind_end_fr, total_conv_stride_over_U1_realized)
    # correction due to stride & inexact unpad indices
    S_2 = (_energy_correction(S_2, B, param_tm, param_fr) if n2 != -1 else
           # `n2=-1` already did time
           _energy_correction(S_2, B, param_fr=param_fr, phi_t_psi_f=True))

    # sanity checks (see "Subsampling, padding") #############################
    if aligned and not global_averaged_fr:
        # `total_conv_stride_over_U1` renamed; comment for searchability
        if scf.__total_conv_stride_over_U1 == -1:
            scf.__total_conv_stride_over_U1 = (  # set if not set yet
                total_conv_stride_over_U1_realized)
        else:
            assert (total_conv_stride_over_U1_realized ==
                    scf.__total_conv_stride_over_U1)

        if not average_fr:
            assert total_conv_stride_over_U1_realized == 0
        elif out_3D:
            max_init_diff = scf.J_pad_frs_max_init - scf.J_pad_frs_max
            expected_common_stride = max(scf.log2_F - max_init_diff -
                                         oversampling_fr, 0)
            assert (total_conv_stride_over_U1_realized ==
                    expected_common_stride)
    stride = (total_conv_stride_over_U1_realized, total_conv_stride_tm)
    return S_2, stride


#### helper methods ##########################################################
def _right_pad(coeff_list, pad_fr, scf, B):
    if scf.pad_mode_fr == 'conj-reflect-zero':
        return _pad_conj_reflect_zero(coeff_list, pad_fr, scf.N_frs_max, B)
    # zero-pad
    zero_row = B.zeros_like(coeff_list[0])
    zero_rows = [zero_row] * (2**pad_fr - len(coeff_list))
    return B.concatenate_v2(coeff_list + zero_rows, axis=1)


def _pad_conj_reflect_zero(coeff_list, pad_fr, N_frs_max, B):
    n_coeffs_input = len(coeff_list)  # == N_fr
    zero_row = B.zeros_like(coeff_list[0])
    padded_len = 2**pad_fr
    # first zero pad, then reflect remainder (including zeros as appropriate)
    n_zeros = min(N_frs_max - n_coeffs_input,  # never need more than this
                  padded_len - n_coeffs_input)    # cannot exceed `padded_len`
    zero_rows = [zero_row] * n_zeros

    coeff_list_new = coeff_list + zero_rows
    right_pad = max((padded_len - n_coeffs_input) // 2, n_zeros)
    left_pad  = padded_len - right_pad - n_coeffs_input

    # right pad
    right_rows = zero_rows
    idx = -2
    reflect = False
    while len(right_rows) < right_pad:
        c = coeff_list_new[idx]
        c = c if reflect else B.conj(c)
        right_rows.append(c)
        if idx in (-1, -len(coeff_list_new)):
            reflect = not reflect
        idx += 1 if reflect else -1

    # (circ-)left pad
    left_rows = []
    idx = - (len(coeff_list_new) - 1)
    reflect = False
    while len(left_rows) < left_pad:
        c = coeff_list_new[idx]
        c = c if reflect else B.conj(c)
        left_rows.append(c)
        if idx in (-1, -len(coeff_list_new)):
            reflect = not reflect
        idx += -1 if reflect else 1
    left_rows = left_rows[::-1]
    return B.concatenate_v2(coeff_list + right_rows + left_rows, axis=1)


def _maybe_unpad_time(Y_2_c, k1_plus_k2, commons2):
    average, log2_T, J, J_pad, N, ind_start, ind_end, unpad, phi = commons2

    start, end = ind_start[0][k1_plus_k2], ind_end[0][k1_plus_k2]
    diff = 0
    if average and log2_T < J:
        min_to_pad = phi['support']
        pad_log2_T = math.ceil(math.log2(N + min_to_pad)) - k1_plus_k2
        padded = J_pad - k1_plus_k2
        N_scale = math.ceil(math.log2(N))
        # need `max` in case `max_pad_factor` makes `J_pad < pad_log2_T`
        # (and thus `padded < pad_log2_T`); need `trim_tm` for indexing later
        diff = max(int(min(padded - pad_log2_T, J_pad - N_scale)), 0)
        if diff > 0:
            Y_2_c = unpad_dyadic(Y_2_c, end - start, 2**J_pad, 2**pad_log2_T,
                                 k1_plus_k2)
    elif not average:
        Y_2_c = unpad(Y_2_c, start, end)
    trim_tm = diff
    return Y_2_c, trim_tm


def _get_stride(j1_fr, pad_fr, subsample_equiv_due_to_pad, scf, average_fr):
    """Actual conv stride may differ due to `oversampling_fr`, so this really
    fetches the reference value. True value computed after convs as
    `n1_fr_subsample + lowpass_subsample_fr`.
    """
    k = subsample_equiv_due_to_pad
    J_pad_frs_max, J_pad_frs_min = scf.J_pad_frs_max, scf.J_pad_frs_min

    if scf.average_fr_global:
        total_conv_stride_over_U1 = min(pad_fr, scf.log2_F)
    elif average_fr:
        phi_fr = scf.phi_f_fr
        if scf.aligned:
            if scf.out_3D:
                k_at_max = scf.J_pad_frs_max_init - J_pad_frs_max
                total_conv_stride_over_U1 = phi_fr['j'][k_at_max]
                if j1_fr is not None:
                    # `None` means we're fetching for unpadding, where `k`
                    # is a reference rather than actual, thus exempt assert
                    assert k == k_at_max
            else:
                k_at_min = scf.J_pad_frs_max_init - J_pad_frs_min
                total_conv_stride_over_U1 = min(phi_fr['j'][k_at_min],
                                                J_pad_frs_min)
        else:
            if scf.out_3D:
                phi_fr_sub_at_max = phi_fr['j'][scf.J_pad_frs_max_init -
                                                J_pad_frs_max]
                total_conv_stride_over_U1 = (
                    min(phi_fr_sub_at_max, J_pad_frs_max) -
                    (J_pad_frs_max - pad_fr)
                )
            else:
                total_conv_stride_over_U1 = phi_fr['j'][k]
    else:
        if scf.aligned:
            total_conv_stride_over_U1 = 0
        else:
            total_conv_stride_over_U1 = j1_fr
    return total_conv_stride_over_U1


def _energy_correction(x, B, param_tm=None, param_fr=None, phi_t_psi_f=False):
    energy_correction_tm, energy_correction_fr = 1, 1
    if param_tm is not None:
        N, ind_start_tm, ind_end_tm, total_conv_stride_tm = param_tm
        # time energy correction due to integer-rounded unpad indices
        unpad_len_exact = N / 2**total_conv_stride_tm
        unpad_len = ind_end_tm - ind_start_tm
        if not (unpad_len_exact.is_integer() and unpad_len == unpad_len_exact):
            energy_correction_tm *= B.sqrt(unpad_len_exact / unpad_len)
        # compensate for subsampling
        energy_correction_tm *= B.sqrt(2**total_conv_stride_tm, dtype=x.dtype)

    if param_fr is not None:
        (N_fr, ind_start_fr, ind_end_fr, total_conv_stride_over_U1_realized
         ) = param_fr
        # freq energy correction due to integer-rounded unpad indices
        unpad_len_exact = N_fr / 2**total_conv_stride_over_U1_realized
        unpad_len = ind_end_fr - ind_start_fr
        if not (unpad_len_exact.is_integer() and unpad_len == unpad_len_exact):
            energy_correction_fr *= B.sqrt(unpad_len_exact / unpad_len,
                                           dtype=x.dtype)
        # compensate for subsampling
        energy_correction_fr *= B.sqrt(2**total_conv_stride_over_U1_realized,
                                       dtype=x.dtype)

        if phi_t_psi_f:
            # since we only did one spin
            energy_correction_fr *= B.sqrt(2, dtype=x.dtype)

    if energy_correction_tm != 1 or energy_correction_fr != 1:
        x *= (energy_correction_tm * energy_correction_fr)
    return x


__all__ = ['timefrequency_scattering1d']
