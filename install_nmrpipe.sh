#!/bin/bash
# Install NMRPipe. Add --no-certificate to the wget commands when behind
# a proxy that decrypts the SSL connection devalidates the certificate.
mkdir bin
cd bin
mkdir NMRPipe
cd NMRPipe
wget https://www.ibbr.umd.edu/nmrpipe/install.com
wget https://www.ibbr.umd.edu/nmrpipe/binval.com
wget https://www.ibbr.umd.edu/nmrpipe/NMRPipeX.tZ
wget https://www.ibbr.umd.edu/nmrpipe/s.tZ
wget https://www.ibbr.umd.edu/nmrpipe/dyn.tZ
wget https://www.ibbr.umd.edu/nmrpipe/talos.tZ
/bin/csh $(pwd)/install.com
#rm -rf install.com binval.com NMRPipeX.tZ s.tZ dyn.tZ talos.tZ
