import os
import sys

sys.path.insert(0, os.path.abspath('../src'))

import pyscad

version = pyscad.__version__
release = pyscad.__version__

project = 'pyscad'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
]

todo_include_todos = True

source_suffix = {
    '.rst':'restructuredtext',
    '.md':'markdown',
}

master_doc = 'index'
language = 'en'

html_theme = 'furo'
