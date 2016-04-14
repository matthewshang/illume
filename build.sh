#!/bin/bash

# nvcc $(find ./src/ -regex ".*\.\(c\|cu\)") -lpng -o ./main.out
nvcc -arch=compute_52 -rdc=true $(find ./src/ -regex ".*\.\(c\|cu\)") -lpng -o illume.o