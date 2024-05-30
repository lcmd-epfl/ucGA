#!/bin/bash

FILE="${1%.*}"

geometry=`cat $FILE.xyz | tail -n +3`

echo "%Mem=8GB
%chk=$FILE
#p wb97xd/6-31g* opt SCF=(maxconventionalcycles=120,xqc) Integral=NoXCTest

best D-A dimers

0 1
$geometry
" > $FILE.com

echo "$FILE xyz file converted to com file"

#to submit

subg16 -p 8 -m 10000 $FILE.com

#for regular GS opts (12-40 GB):
#p wb97xd/6-31g* opt freq=noraman SCF=(maxconventionalcycles=120,xqc) Integral=NoXCTest

#for LOOSE GS opts:
#p wb97xd/6-31G* opt=loose freq=noraman SCF=(conver=6,maxconventionalcycles=120,xqc) Integral=NoXCTest

#for vertical excitations with extra output for THEODORE (40 GB):
#p wB97XD/6-31g* tda=(nstates=5,50-50) scf=(maxconventionalcycles=120,xqc) Integral=NoXCTest pop=full iop(9/40=3) GFINPUT 
#p wB97XD/6-31g* nosymm tda=(nstates=5,50-50) scf=(maxconventionalcycles=120,xqc) Integral=NoXCTest pop=full iop(9/40=3) GFINPUT

#for S1 opts:
#p wb97xd/6-31g* opt tda=(nstates=3,root=1,singlets) scf=(maxconventionalcycles=120,xqc) Integral=NoXCTest 

#for T1 opts:
#p wb97xd/6-31g* opt tda=(nstates=3,root=1,triplets) scf=(maxconventionalcycles=120,xqc) Integral=NoXCTest 

#for T2 opts:
#p wb97xd/6-31g* opt tda=(nstates=3,root=2,triplets) scf=(maxconventionalcycles=120,xqc) Integral=NoXCTest

#for loose S1 opts:
#p wb97xd/6-31g* opt=loose nosymm tda=(nstates=3,root=1,singlets) scf=(conver=6,maxconventionalcycles=120,xqc) Integral=NoXCTest

#for loose T1 opts:
#p wb97xd/6-31g* opt=loose nosymm tda=(nstates=3,root=1,triplets) scf=(conver=6,maxconventionalcycles=120,xqc) Integral=NoXCTest

