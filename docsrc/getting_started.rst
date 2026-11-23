###############
Getting started
###############

First analysis
==============

The repository ships a six-month, 15-minute discharge record from Freeport
(``examples/freeport/freeport_2014_select.csv``, 18001 samples, m\ :sup:`3`/s). It is
a dms_datastore-style CSV: ``#`` header lines followed by a ``datetime,value`` table.

.. code-block:: bash

   scha fit examples/freeport/freeport_2014_select.csv \
       -o fit.yaml --decomposition components.csv

This takes about half a minute and logs progress and a summary::

   SCHA(2,3) fit, method=constrained, step=15min, bands=D1,D2,D3,D4,D5,D6
     data: 2014-02-22 01:45:00 to 2014-08-28 13:45:00, n=18001, observed=18001 (0.0% missing)
     loglik=-59971.11  success=True  Optimization terminated successfully
     rho=0.989628  sigma2_eps=2.5350e+01  sigma2_zeta=3.8400e-03
     sigma2_kappa: D1=1.238e-06  D2=8.619e-07  D3=1.390e-07  D4=5.873e-08  D5=1.134e-08  D6=4.087e-09
     subtide/D1 crossover: 0.589 cpd
     requirements: D1 min 0.9951  D2 min 0.9997  D3 min 0.9992  D4 min 0.9980  sub*H max 1.00e-02@1.41

The variances carry the units of the data squared; ``rho`` and the gains do not.
The times in this example file are nominal (see the ``time_note`` in its header), so
absolute phases from it are nominal too.

``fit.yaml`` holds the fitted parameters and diagnostics. ``components.csv`` holds
the smoothed decomposition of the series.

Input files
-----------

Input files are CSV with the time in the first column (datetimes, or numeric hours
since ``--reftime``) and the values in the next column (choose another with
``--column``). Lines beginning with ``#`` are skipped. Times must be naive (no time
zone) and on a regular step, with missing rows allowed; the step is the most common
spacing unless ``--freq`` (for example ``15min``) is given. Missing times and
non-numeric values become gaps.

By default ``scha fit`` uses the *constrained* method: maximum likelihood subject
to requirements on the steady-state filter gains, which keep each band on its own
tidal lines and keep the subtide out of the tidal bands. See :doc:`advanced`.

Output options and post-processing
==================================

Every command logs progress to the console. Common options are ``--logdir`` (also
write a log file), ``--debug`` and ``-q/--quiet``. Use ``scha <command> -h`` for the
options of each command; the full list is in :doc:`commands`.

Once a fit exists it can be applied to the data without refitting:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Command
     - Output
   * - ``scha decompose series.csv --fit fit.yaml -o components.csv``
     - ``observed``, ``fitted``, ``subtide``, ``subtide_raw``, one column per band,
       ``residual``
   * - ``scha phasors series.csv --fit fit.yaml -o phasors.csv``
     - ``<band>_c`` and ``<band>_s``: the smoothed top-pair states of each band
   * - ``scha ampphase series.csv --fit fit.yaml -o ampphase.csv``
     - ``<band>_a``, ``<band>_b`` (local cos and sin coefficients), ``<band>_amp``
       and ``<band>_phase``
   * - ``scha gains fit.yaml -o gains.csv``
     - steady-state filter gains by frequency (cpd): ``subtide``, one column per
       band, ``noise`` and ``subtide_filtered``

Notes on the products:

* The decomposition satisfies ``residual = observed - subtide - sum(bands)``. The
  default 18 h cosine-Lanczos post-filter on the subtide leaves ``subtide`` undefined
  (NaN) near the ends of the record; ``subtide_raw`` is the unfiltered smoother
  output, and the part removed by the filter ends up in ``residual``. Use
  ``--post-filter none`` to turn the post-filter off.
* Phase is the lag in degrees of ``A cos(w t - phase)``, with ``t`` in hours since an
  epoch (default 1970-01-01, set with ``--epoch``). Each band is referenced to its
  model line (``k`` times M1); use ``--ref D1=K1`` to reference a band to a named
  constituent instead.
* The ``fitted`` column of the decomposition is the smoothed signal at every time
  step, including gaps.
* A fit applies to data with the same time step it was fitted at (stored as
  ``fit_freq`` in the YAML). Apply it to other records of that step, for example
  another year at the same station, with the ``--fit`` option.
