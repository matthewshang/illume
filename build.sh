#!/bin/bash

start=$(date +%s%N)
nvcc -arch=compute_52 -rdc=true $(find ./src/ -regex ".*\.\(c\|cu\)") -lpng -o illume.o
tt=$((($(date +%s%N) - $start)/1000000000))
echo "Build time: $tt seconds"