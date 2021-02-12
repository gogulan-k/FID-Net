#!/bin/csh

#
# Final processing of 13C dimension after decoupling

 xyz2pipe -in test_decouple.ft2 -z -verb \
| nmrPipe  -fn SP -off 0.5 -end 0.98 -pow 2 -c 0.5  \
| nmrPipe  -fn ZF -auto                             \
| nmrPipe  -fn FT -alt                                 \
| nmrPipe  -fn PS -p0 0.0 -p1 0.0 -di               \
    -ov -out test_decouple.ft3

