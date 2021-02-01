#!/usr/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

ZIPPATH="${SCRIPTPATH}/shapenetcore_partanno_segmentation_benchmark_v0.zip"


wget --no-check-certificate  https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip -P $SCRIPTPATH
unzip $ZIPPATH -d $SCRIPTPATH
