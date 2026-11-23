# Sphinx configuration for scha
import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

# Try import the package to get version
try:
    import scha
    version = scha.__version__
    release = version
except Exception:
    version = '0+unknown'
    release = version

project = 'scha'
copyright = '2026, California Department of Water Resources'
author = 'Eli Ateljevich'

needs_sphinx = '8.2'  # sphinx.ext.apidoc

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.apidoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx_click',
    #'sphinx_autodoc_typehints',
    'sphinxcontrib.bibtex',
    'numpydoc',
]

# API pages are generated into the ignored docsrc/_api on every build. Objects are
# documented once, under the module that defines them; the package-level re-exports
# in scha/__init__.py are not repeated on the scha page.
apidoc_modules = [
    {
        'path': '../src/scha',
        'destination': '_api',
        'automodule_options': {'members', 'undoc-members', 'show-inheritance', 'ignore-module-all'},
    }
]

autosummary_generate = True
autosummary_imported_members = False
numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False

templates_path = ['_templates']
master_doc = 'index'
exclude_patterns = ['_build', '**.ipynb_checkpoints']

# Bibliography
bibtex_bibfiles = ['references.bib']

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable', None),
}

html_theme = 'sphinx_rtd_theme'
html_logo = '_static/SCHA_logo.png'
html_theme_options = {'logo_only': False}
html_static_path = ['_static']
html_domain_indices = ['py-modindex']

# small helper to ignore test functions in autodoc output

def autodoc_skip_member_handler(app, what, name, obj, skip, options):
    if name.startswith('test_'):
        return True
    return None


def setup(app):
    app.connect('autodoc-skip-member', autodoc_skip_member_handler)
