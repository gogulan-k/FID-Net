#!/bin/csh

#
# Basic 2D Phase-Sensitive Processing:
#   Cosine-Bells are used in both dimensions.
#   Use of "ZF -auto" doubles size, then rounds to power of 2.
#   Use of "FT -auto" chooses correct Transform mode.
#   Imaginaries are deleted with "-di" in each dimension.
#   Phase corrections should be inserted by hand.

nmrPipe -in dl.ft1 \
| nmrPipe  -fn TP                                     \
#| nmrPipe -fn LP -fb -ord 8                                         \
| nmrPipe  -fn SP -off 0.5  -pow 2 -c 1.0    \
| nmrPipe  -fn ZF -auto                               \
| nmrPipe  -fn FT -alt                                \
| nmrPipe  -fn PS -p0 -90.00 -p1 180.00 -di -verb         \
#| nmrPipe -fn POLY -auto \
# | nmrPipe -fn EXT -x1 8.0ppm -xn 30.0ppm -sw \
   -ov -out dl.ft2

rescale.com -in dl.ft2 -out dl_rescale.ft2 -max 1.0
