#!/bin/bash
for f in evo/prev_evos/evo_2023040*/prev_gens evo/curr*/*/prev_gens; do
    echo -n $f
    gencount=0
    for g in $f/gen*/scores.csv; do
        gencount=$(( gencount + 1 )); 
    done
    echo ": "$gencount
done
