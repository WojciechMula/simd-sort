================================================================================
                                SIMD sorting
================================================================================

Overview
--------------------------------------------------

This repository contains SIMD versions of quicksort, using **AVX512F**
and **AVX2** instruction sets. The subdirectory ``results`` contains
performance results.

The code is still unfinished, it may be subject of changes.

Altrough programs are written in C++, most procedures are nearly
plain C, except use of namespaces and references in place of
pointers.


Building
--------------------------------------------------

At least GCC 5.3.0 is needed.  Type ``make`` to build AVX512F
and AVX2 versions of two programs:

* ``test`` --- tests if SIMD-ized sorting procedures work correctly
  for various inputs.

* ``speed`` --- compares speed of different procedures, scalar and
  SIMD for various inputs.


AVX512
--------------------------------------------------

`Intel Software Development Emulator`__ can be used to run AVX512 variants.

__ https://software.intel.com/en-us/articles/intel-software-development-emulator

