################
Programmatic use
################

Each command has the same capability in Python, so the command line and scripts give
the same results:

.. code-block:: python

   import scha
   from scha.io import read_series

   y = read_series("examples/freeport/freeport_2014_select.csv")  # regular pd.Series, NaN for gaps

   fit = scha.fit(y)                            # SCHA(2,3), D1-D6, constrained
   print(fit.summary())

   dec = fit.decompose()                        # observed, fitted, subtide, D1.., residual
   ap = scha.amplitude_phase(fit.phasors())     # D1_a, D1_b, D1_amp, D1_phase, ...
   gains = fit.gains()                          # gain by frequency (cpd)
   filled = fit.fill()                          # y with gaps filled

   fit.to_yaml("fit.yaml")
   fit2 = scha.load_fit("fit.yaml")
   dec2 = fit2.decompose(y)                     # a loaded fit needs the data

``y`` must be a ``pandas.Series`` with a naive ``DatetimeIndex`` whose ``freq`` is set,
and NaN where there are gaps. :func:`scha.io.read_series` and
:func:`scha.io.regularize` build one from a CSV file or an irregular series.

Useful methods of the returned :class:`scha.fit.SCHAFit`:

* ``fit.summary()``: parameters, likelihood and requirement report.
* ``fit.decompose(y)``, ``fit.phasors(y)``, ``fit.fill(y)``: products of the smoother.
* ``fit.innovations(y)``: standardized one-step prediction errors, a diagnostic of
  model adequacy.
* ``fit.gains(cpd)``: steady-state gains of the subtide, each band and the noise.
* ``fit.crossover()``: the frequency at which the subtide and D1 gains cross.

The work functions behind the commands are in :mod:`scha.cli` (:func:`scha.cli.fit_file`,
:func:`scha.cli.decompose_file`, :func:`scha.cli.phasors_file`,
:func:`scha.cli.ampphase_file`, :func:`scha.cli.gains_file` and
:func:`scha.cli.crossval_file`). They take file names and perform the same steps as
the command line.
