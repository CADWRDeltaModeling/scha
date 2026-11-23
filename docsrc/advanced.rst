##############
Advanced usage
##############

Bands and model order
=====================

``--bands`` accepts a range (``D1-D8``) or a list (``D1,D2,D4``); ``--order m,n``
sets the trend and cycle order (for example ``2,4``). In Python these are
``bands=`` and ``order=(m, n)``. Higher bands absorb overtides and compound tides.
By default the highest two fitted bands are *sacrificial*: they carry no passband
requirement and exist to keep high-frequency content away from the subtide, the
noise and the main bands.

Gain requirements
=================

The constrained method keeps each band on its tidal lines through requirements
stated as steady-state smoother gains. In Python they are given as a
:class:`scha.requirements.Requirements` object (or a dict):

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Python (default)
     - CLI flag
     - Meaning
   * - ``delta`` (0.01)
     - ``--delta``
     - each band's gain is at least ``1 - delta`` on its named tidal lines
   * - ``eps`` (0.01)
     - ``--eps``
     - subtide gain is at most ``eps`` at the gaps between bands and above the last
       band
   * - ``sacrificial`` (2)
     - ``--sacrificial``
     - bands without a passband requirement: a count of the highest bands, or a spec
       such as ``D5-D6``
   * - ``sub_cross``
     - ``--sub-cross``
     - require the subtide gain to be at least the D1 gain at this frequency (cpd)
   * - ``sub_pass``
     - ``--sub-pass``
     - require a subtide passband up to this frequency (cpd)
   * - ``post_filter`` (``"18h"``)
     - ``--post-filter``
     - cosine-Lanczos cutoff applied to the subtide; ``none`` turns it off

For example:

.. code-block:: bash

   scha fit series.csv -o fit.yaml --bands D1-D8 --sacrificial D5-D8 --delta 0.02

The requirement report in ``fit.summary()`` lists the achieved minimum passband gains.
Very demanding requirements can be infeasible: a subtide passband reaching far above
the lowest tidal band cannot be met at realistic values of ``rho``, and a crossover
raised in one large step can land in a poor local optimum. Raise such requirements in
small steps, each started from the previous fit (``--start fit.yaml``).

Methods, fixed parameters and starting values
=============================================

``--method`` (``method=``) selects the estimator:

* ``constrained`` (default): maximum likelihood under the gain requirements.
* ``ml``: plain maximum likelihood, with the requirements only reported.
* ``em``: EM with ``rho`` held fixed; no requirements, and ``sigma2_kappa`` cannot be
  held fixed.

Parameters can be held fixed with ``--fix`` (repeatable) or ``fixed=``:

.. code-block:: bash

   scha fit series.csv -o fit.yaml --fix rho=0.985 --fix sigma2_kappa:D1=1e-10

The fixable parameters are ``rho``, ``sigma2_eps``, ``sigma2_zeta`` and the band
variances ``sigma2_kappa``. Starting values are chosen automatically from the
periodogram; use ``--start other_fit.yaml`` (``start=``) to start from a previous fit.

Configuration files
===================

The keys of ``--config`` are the keyword arguments of :func:`scha.cli.fit_file`
(``column``, ``freq``, ``reftime``, ``bands``, ``order``, ``method``,
``requirements``, ``fixed``, ``start``, ``maxiter``, ``workers``, ``decomposition``).
Command-line flags override the file. In YAML, write ``null`` for "no post-filter":

.. code-block:: yaml

   bands: D1-D6
   order: [2, 3]
   requirements:
     delta: 0.02
     post_filter: null
   fixed:
     rho: 0.985

Cross-validation
================

``scha crossval series.csv --config cv.yaml -o cv.csv`` scores a grid of ``rho`` and
``sigma2_zeta`` values by predicting withheld blocks of data. The config must list
``rho`` and ``sigma2_zeta`` values; the optional keys are ``bands``, ``order``,
``gap``, ``spacing``, ``first``, ``refit_folds``, ``requirements``, ``maxiter``,
``workers``, ``column``, ``freq`` and ``reftime`` (see :func:`scha.cli.crossval_file`).

Lower-level tools
=================

The pieces behind :func:`scha.fit.fit` are importable for research use:

* :func:`scha.mlfit.fit_ml`: constrained and unconstrained maximum likelihood.
* :mod:`scha.fastll`: numba filter for the likelihood (``scha_loglik``,
  ``scha_loglik_concentrated``), standardized innovations and smoothed states.
* :mod:`scha.ffbs`: reference Kalman filter, smoother and Carter-Kohn draw.
* :func:`scha.em_fit.scha_em`: EM estimation.
* :mod:`scha.gains`: closed-form component spectra and gains.
* :func:`scha.state_space.build_scha_state_space` and :class:`scha.params.SCHAParams`: the
  model matrices.

Internally the state is scaled so that cycles with ``rho`` near 1 remain well
conditioned; the transformation is exact.
