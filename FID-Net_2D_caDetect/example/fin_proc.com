#!/bin/csh

#
# Basic 2D Phase-Sensitive Processing:
#   Cosine-Bells are used in both dimensions.
#   Use of "ZF -auto" doubles size, then rounds to power of 2.
#   Use of "FT -auto" chooses correct Transform mode.
#   Imaginaries are deleted with "-di" in each dimension.
#   Phase corrections should be inserted by hand.

nmrPipe -in decouple.ft1 \
| nmrPipe  -fn SP -off 0.45 -end 0.98 -pow 2 -c 0.5    \
| nmrPipe  -fn ZF -auto                               \
| nmrPipe  -fn FT -auto                              \
| nmrPipe  -fn PS -p0 -9.0 -p1 29.00 -di -verb         \
   -ov -out decouple.ft2
