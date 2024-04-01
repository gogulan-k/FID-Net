#!/bin/csh

nmrPipe -in decouple.ft1 \
| nmrPipe  -fn SP -off 0.48 -end 0.98  -pow 1 -c 0.5    \
| nmrPipe  -fn ZF -auto \                       \
| nmrPipe  -fn FT -auto                                \
| nmrPipe  -fn PS -p0 0.00 -p1 0.00 -di -verb         \
| nmrPipe  -fn TP \
   -ov -out decouple.ft2
