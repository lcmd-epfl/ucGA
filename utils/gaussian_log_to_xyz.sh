#!/bin/bash

#module load openbabel/2.3.2/gcc-4.4.7
module load openbabel/2.4.1/gcc-4.8.5

shopt -s nullglob #use this line in for loop going through list, otherwise mysterious problems in openbabel

export FILE=${1%.*} 

if grep -q 'Normal termination' ${FILE}.log; then

 if ! grep -q 'imaginary frequencies (negative Signs)' ${FILE}.log; then
  echo "Making XYZ from $FILE"
  obabel -ig09 ${FILE}.log -oxyz -O ${FILE}_relaxed.xyz #TO GENERATE xyz from log; DO NOT ADD '--gen3d' (does a forcefield optimization)

 else echo "$FILE has negative frequencies!"
 fi

else echo "$FILE did not terminate normally!"
fi

