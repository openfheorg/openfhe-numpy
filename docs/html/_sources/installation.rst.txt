Installation
============

OpenFHE-Numpy requires the OpenFHE library to be installed first.

Prerequisites
-------------

- Python 3.8+
- OpenFHE library (version 1.3.1+)
- NumPy
- CMake (for building from source)

Installing OpenFHE
-------------------

First, install the OpenFHE library. Please refer to the
`OpenFHE installation guide <https://github.com/openfheorg/openfhe-development>`_.

Installing OpenFHE-Numpy
-------------------------

From Source
~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/openfheorg/openfhe-numpy.git
   cd openfhe-numpy

   # Create build directory
   mkdir build && cd build

   # Configure with CMake
   cmake ..

   # Build and install
   make
   sudo make install

Using pip (when available)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install openfhe-numpy

Verification
------------

To verify your installation:

.. code-block:: python

   import openfhe_numpy as onp
   print(onp.__version__)

Development Installation
------------------------

For development, create a virtual environment:

.. code-block:: bash

   python3 -m venv openfhe-env
   source openfhe-env/bin/activate  # Linux/macOS
   pip install -e .

This installs OpenFHE-Numpy in editable mode for development.
