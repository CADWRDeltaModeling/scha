.. scha documentation master file

###########
Quick Intro
###########

``scha`` is a Python implementation of stochastic cycle harmonic analysis (SCHA).
It fits a state-space model of a tidal time series: a smooth trend (the *subtide*),
plus higher-order stochastic cycles at the tidal harmonics (the *bands* D1, D2, ...),
plus noise. Parameters are estimated by maximum likelihood with a Kalman filter,
and the components are recovered with a Kalman smoother. Because the cycles are
stochastic, the amplitude and phase of each band may vary slowly in time, so the
tide need not be stationary, and gaps in the record are handled naturally.

The method is described in Ateljevich and Wang (2023), *Nonstationary analysis of
tides using higher order stochastic cycles*. ``scha`` is the Python counterpart of
the R package ``stochCycle``.

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   self
   install
   getting_started
   programmatic
   advanced
   commands
   examples
   reference
   API Reference <_api/modules>
   contributing_docs

The model
=========

.. code-block:: text

   y_t = trend_t + sum_j cycle_j,t + eps_t

The trend is an m-th order integrated random walk. Each cycle is an n-th order
stochastic cycle with damping ``rho`` at a fixed frequency, a multiple of the M1
frequency (about 0.966 cpd). ``SCHA(m,n)`` names the orders. The default is
SCHA(2,3) with bands D1 to D6.

Quickstart
==========

Fit a series from the command line, then use the stored fit:

.. code-block:: bash

   scha fit series.csv -o fit.yaml --decomposition components.csv
   scha ampphase series.csv --fit fit.yaml -o ampphase.csv

or in Python:

.. code-block:: python

   import scha
   from scha.io import read_series

   y = read_series("series.csv")
   fit = scha.fit(y)
   print(fit.summary())
   dec = fit.decompose()

See :doc:`getting_started` for a worked example.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
