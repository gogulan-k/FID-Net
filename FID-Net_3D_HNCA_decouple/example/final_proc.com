#!/bin/csh

#
# 3D States-Mode HN-Detected Processing.

 xyz2pipe -in test_decouple.ft2 -z -verb \
#nmrPipe -in T4L_decoup.ft2   -verb               \
#| nmrPipe -fn TP	\
#| nmrPipe -fn ZTP	\
| nmrPipe  -fn SP -off 0.5 -end 0.98 -pow 2 -c 0.5  \
| nmrPipe  -fn ZF -auto                             \
| nmrPipe  -fn FT -alt                                 \
| nmrPipe  -fn PS -p0 0.0 -p1 0.0 -di               \
    -ov -out test_decouple.ft3

