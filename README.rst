================================================================================
                                SIMD sorting
================================================================================

Overview
--------------------------------------------------

This repository contains SIMD versions of quicksort, using **AVX512F**
and **AVX2** instruction sets. The subdirectory ``results`` contains
performance results.

The code is still unfinished, it may be subject of changes.

Although programs are written in C++, most procedures are nearly
plain C, except use of namespaces and references in place of
pointers.


Building
--------------------------------------------------

At least GCC 5.3.0 is needed.  Type ``make`` to build AVX512F
and AVX2 versions of two programs:

* ``test``/``test_avx2`` --- tests if SIMD-ized sorting procedures
  work correctly for various inputs.

* ``speed``/``speed_avx2`` --- compares speed of different procedures,
  scalar and SIMD for various inputs.

* ``speed_stats``/``speed_avx2_stats`` --- similar to ``speed``, but
  some procedures collect and print runtime statistics. Note that
  this gives overhead, so for real performance tests the basic variant
  of the programs should be used.


AVX512
--------------------------------------------------

`Intel Software Development Emulator`__ can be used to run AVX512 variants.

__ https://software.intel.com/en-us/articles/intel-software-development-emulator


References
--------------------------------------------------

- Fast Sorting Algorithms using AVX-512 on Intel Knights Landing https://arxiv.org/pdf/1704.08579.pdf
