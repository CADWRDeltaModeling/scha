===================================================================
scha: A Python Implementation of Stochastic Cycle Harmonic Analysis
===================================================================

``scha`` fits a state-space model of a tidal time series: a smooth trend (the
subtide), plus higher-order stochastic cycles at the tidal harmonics, plus noise.
Parameters are estimated by maximum likelihood with a Kalman filter, and the
components are recovered with a Kalman smoother, so the amplitude and phase of each
tidal band may vary in time and gaps in the record are handled naturally. The method
is described in Ateljevich and Wang (2023), *Nonstationary analysis of tides using
higher order stochastic cycles*.

**Full documentation:** https://cadwrdeltamodeling.github.io/scha/

Documentation sources are in ``docsrc/``; build them with ``make html`` (see
``docsrc/contributing_docs.rst``).

Installation
============

.. code-block:: bash

   git clone https://github.com/CADWRDeltaModeling/scha.git
   cd scha
   pip install -e .

``scha`` requires numpy, scipy, pandas, numba, pyyaml, click and ``vtools3``.

Quick example
=============

.. code-block:: bash

   scha fit examples/freeport/freeport_2014_select.csv \
       -o fit.yaml --decomposition components.csv
