#!/bin/bash

# Run GPNH-regularized convex coding on PCs of JRA-55 500 hPa height anomalies with
# standard settings for algorithm parameters.
#
# Usage: run_jra55_pca_gpnh_wrapper.sh <n_components> <lambda_W>
#
# License: MIT

BIN_DIR=$(dirname $0)
ABSBASEDIR=$(cd "$BIN_DIR/.."; pwd)

DATA_DIR="${ABSBASEDIR}/data"
BASE_RESULTS_DIR="${ABSBASEDIR}/results"

if test ! -d "$BASE_RESULTS_DIR" ; then
   mkdir -p "$BASE_RESULTS_DIR"
fi

HGT_RESULTS_DIR="${BASE_RESULTS_DIR}/jra55"

if test ! -d "$HGT_RESULTS_DIR" ; then
   mkdir -p "$HGT_RESULTS_DIR"
fi

RESULTS_DIR="${HGT_RESULTS_DIR}/nc"

if test ! -d "$RESULTS_DIR" ; then
   mkdir -p "$RESULTS_DIR"
fi

PYTHON="python"
RUN_GPNH="${BIN_DIR}/run_jra55_pca_gpnh.py"

BASE_PERIOD_START_YEAR="1981"
BASE_PERIOD_END_YEAR="2010"

LAT_WEIGHTS="scos"
N_EOFS="167"
RANDOM_SEED="0"
INIT="random"
N_INIT="100"
MAX_ITERATIONS="10000"
TOLERANCE="1e-6"
RESTRICT_TO_BASE_PERIOD="no"
VERBOSE="yes"
STANDARDIZE="no"

if test $# -ne 2 ; then
   echo "Error: too few arguments"
   echo "Usage: `basename $0` <n_components> <lambda_W>"
   exit 1
fi

n_components="$1"
lambda_W="$2"

opts="\
--n-components $n_components \
--lambda-W $lambda_W \
--init $INIT \
--n-init $N_INIT \
--tolerance $TOLERANCE \
--max-iterations $MAX_ITERATIONS \
--random-seed $RANDOM_SEED \
"

base_filename="${RESULTS_DIR}/jra.55.hgt.500.1958010100_2018123118.anom.${BASE_PERIOD_START_YEAR}_${BASE_PERIOD_END_YEAR}.ALL.pca.${LAT_WEIGHTS}.k${N_EOFS}"

input_file="${base_filename}.nc"

if test "x$STANDARDIZE" = "xyes" ; then
   output_file="${base_filename}.gpnh.k${n_components}.lambda_W$(printf %5.3e $lambda_W).standardized.n_init${N_INIT}.nc"
   opts="$opts --standardize"
else
   output_file="${base_filename}.gpnh.k${n_components}.lambda_W$(printf %5.3e $lambda_W).n_init${N_INIT}.nc"
fi

if test "x$RESTRICT_TO_BASE_PERIOD" = "xyes" ; then
   opts="$opts --restrict-to-base-period"
fi

if test "x$VERBOSE" = "xyes" ; then
   opts="$opts --verbose"
fi

$PYTHON "$RUN_GPNH" $opts "$input_file" "$output_file"
