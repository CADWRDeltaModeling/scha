############
Installation
############

``scha`` requires Python below 3.14 and the usual scientific stack: numpy, scipy,
pandas, numba, pyyaml and click. It also uses ``vtools3`` (imported as ``vtools``)
for the cosine-Lanczos filter that smooths the subtide.

.. note::

   The distribution name on PyPI is ``vtools3``. The PyPI package called ``vtools``
   is unrelated.

Install from a clone of the repository into a conda environment that has the
dependencies:

.. code-block:: bash

   git clone https://github.com/CADWRDeltaModeling/scha.git
   cd scha
   pip install -e .

or, without a clone:

.. code-block:: bash

   pip install git+https://github.com/CADWRDeltaModeling/scha.git

If the dependencies are already installed (for example from conda), use
``pip install --no-deps -e .``.

The install provides the ``scha`` command; check it with ``scha -h``. For
development, add the test tools with ``pip install -e .[dev]`` and run
``pytest tests``.

The first call to a fitting function compiles the numba filters, which takes a
few seconds.

The ``scha`` command and the Python API are the same capability in two forms; see
:doc:`getting_started` and :doc:`programmatic`.
