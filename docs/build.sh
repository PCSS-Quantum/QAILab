#!/bin/bash

# Builds the documentation 

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

cp ./../tutorials/*.ipynb ./tutorials

make clean
make html