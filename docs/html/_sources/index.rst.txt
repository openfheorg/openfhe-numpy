.. OpenFHE-Numpy documentation master file, created by
   sphinx-quickstart on Mon Aug 11 14:51:10 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

OpenFHE-Numpy Documentation
============================

Welcome to OpenFHE-Numpy! This library provides NumPy-compatible matrix operations
that can be performed on encrypted data using homomorphic encryption.

OpenFHE-Numpy allows you to perform mathematical operations on encrypted data without
decrypting it, maintaining privacy while enabling computation on sensitive information.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/modules
   api/direct_modules

Features
========

* **NumPy Compatibility**: Familiar NumPy-like API for encrypted computations
* **Homomorphic Encryption**: Perform operations on encrypted data
* **Matrix Operations**: Support for matrix arithmetic, linear algebra, and reductions
* **Type Safety**: Full type hints and comprehensive documentation

Quick Example
=============

.. code-block:: python

   import numpy as np
   import openfhe_numpy as onp
   from openfhe import *

   # Initialize CKKS context
   params = CCParamsCKKSRNS()
   params.SetMultiplicativeDepth(7)
   params.SetScalingModSize(59)
   params.SetFirstModSize(60)
   params.SetScalingTechnique(FIXEDAUTO)
   params.SetSecretKeyDist(UNIFORM_TERNARY)

   cc = GenCryptoContext(params)
   cc.Enable(PKESchemeFeature.PKE)
   cc.Enable(PKESchemeFeature.LEVELEDSHE)
   cc.Enable(PKESchemeFeature.ADVANCEDSHE)

   # Generate keys
   keys = cc.KeyGen()
   cc.EvalMultKeyGen(keys.secretKey)
   cc.EvalSumKeyGen(keys.secretKey)

   # Create matrix and encrypt it
   A = np.array([[1, 2], [3, 4]])

   ring_dim = cc.GetRingDimension()
   total_slots = ring_dim // 2

   # Encrypt with OpenFHE-NumPy
   ctm_A = onp.array(
         cc=cc,
         data=A,
         batch_size=batch_size,
         order=onp.ROW_MAJOR,
         fhe_type="C",
         mode="zero",
         public_key=keys.publicKey,
      )


   # Generate keys
   onp.EvalSquareMatMultRotateKeyGen(keys.secretKey, ctm_A.ncols)

   # Perform encrypted operations
   ctm_product = ctm_A @ ctm_A      # Matrix multiplication
   ctm_sum = onp.add(ctm_A, ctm_A)  # Element-wise addition

   # Decrypt results
   decrypted_product = ctm_product.decrypt(keys.secretKey, unpack_type="original")
   decrypted_sum = ctm_sum.decrypt(keys.secretKey, unpack_type="original")

   print("Result of A @ A:")
   print(decrypted_product)

   print("Result of A + A:")
   print(decrypted_sum)

API Reference
=============

.. autosummary::
   :toctree: api/
   :recursive:

   openfhe_numpy

Core Operations
===============

.. autosummary::
   :toctree: api/

   openfhe_numpy.operations.matrix_api.add
   openfhe_numpy.operations.matrix_api.subtract
   openfhe_numpy.operations.matrix_api.multiply
   openfhe_numpy.operations.matrix_api.matmul
   openfhe_numpy.operations.matrix_api.dot
   openfhe_numpy.operations.matrix_api.sum
   openfhe_numpy.operations.matrix_api.mean
   openfhe_numpy.operations.matrix_api.cumulative_sum
   openfhe_numpy.operations.matrix_api.transpose
   openfhe_numpy.operations.matrix_api.roll

Tensor Classes
==============

.. autosummary::
   :toctree: api/

   openfhe_numpy.tensor.tensor.Tensor
   openfhe_numpy.tensor.ctarray.CTArray

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
