Contributing To The Docs
========================

Edit the reStructuredText (``.rst``) sources in ``docsrc/``. Generated HTML is not
committed.

Environment
-----------

Use the ``schism`` conda environment with ``scha`` installed, and add the
documentation tools to it (Sphinx 8.2 or later is required for
``sphinx.ext.apidoc``):

.. code-block:: console

   conda activate schism
   pip install --no-deps -e .
   pip install -r docsrc/docs-requirements.txt

Build And Preview
-----------------

From ``docsrc``, run ``make html`` (``.\make.bat html`` in Windows PowerShell). For a
clean rebuild, run ``make clean`` followed by ``make html``.

.. code-block:: console

   cd docsrc
   make html

``sphinx.ext.apidoc`` generates the API pages into ``docsrc/_api/`` on every build;
they are ignored by git, so do not edit or commit them. HTML goes to the ignored
``docs/html``; open ``docs/html/index.html`` and inspect the pages you changed. Add new
pages to the ``toctree`` in ``docsrc/index.rst``.
