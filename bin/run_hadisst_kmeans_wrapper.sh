#!/bin/bash

# Run k-means clustering on HadISST SST anomalies with
# standard settings for algorithm parameters.
#
# Usage: run_hadisst_kmeans_wrapper.sh <n_clusters>
#
# License: MIT

BIN_DIR=$(dirname $0)
ABSBASEDIR=$(cd "$BIN_DIR/.."; pwd)

DATA_DIR="${ABSBASEDIR}/data"
BASE_RESULTS_DIR="${ABSBASEDIR}/results"

if test ! -d "$BASE_RESULTS_DIR" ; then
   mkdir -p "$BASE_RESULTS_DIR"
fi

SST_RESULTS_DIR="${BASE_RESULTS_DIR}/hadisst"

if test ! -d "$SST_RESULTS_DIR" ; then
   mkdir -p "$SST_RESULTS_DIR"
fi

RESULTS_DIR="${SST_RESULTS_DIR}/nc"

if test ! -d "$RESULTS_DIR" ; then
   mkdir -p "$RESULTS_DIR"
fi

PYTHON="python"
RUN_KMEANS="${BIN_DIR}/run_hadisst_kmeans.py"

BASE_PERIOD_START_YEAR="1981"
BASE_PERIOD_END_YEAR="2010"
ANOMALY_TREND_ORDER="1"

LAT_WEIGHTS="scos"
RANDOM_SEED="0"
INIT="k-means++"
N_INIT="100"
REFERENCE="uniform"
N_TRIALS="100"
MAX_ITERATIONS="10000"
TOLERANCE="1e-4"
N_JOBS="1"

STANDARDIZED="no"

CROSS_VALIDATE="no"
N_FOLDS="10"

RESTRICT_TO_BASE_PERIOD="no"
VERBOSE="no"

if test $# -ne 1 ; then
   echo "Error: too few arguments"
   echo "Usage: `basename $0` <n_clusters>"
   exit 1
fi

n_components="$1"

opts="\
--n-components $n_components \
--lat-weights $LAT_WEIGHTS \
--init $INIT \
--n-init $N_INIT \
--tolerance $TOLERANCE \
--max-iterations $MAX_ITERATIONS \
--random-seed $RANDOM_SEED \
--n-trials $N_TRIALS \
--n-jobs $N_JOBS \
--reference $REFERENCE \
"

if test "x$STANDARDIZED" = "xyes" ; then
   base_filename="${RESULTS_DIR}/HadISST_sst.std_anom.${BASE_PERIOD_START_YEAR}_${BASE_PERIOD_END_YEAR}.trend_order${ANOMALY_TREND_ORDER}"
   opts="$opts --standardized"
else
   base_filename="${RESULTS_DIR}/HadISST_sst.anom.${BASE_PERIOD_START_YEAR}_${BASE_PERIOD_END_YEAR}.trend_order${ANOMALY_TREND_ORDER}"
fi

input_file="${base_filename}.nc"

if test "x$CROSS_VALIDATE" = "xyes" ; then
   output_file="${base_filename}.kmeans.${LAT_WEIGHTS}.k${n_components}.n_init${N_INIT}.${REFERENCE}_gap.n_folds${N_FOLDS}.nc"
   opts="$opts --cross-validate --n-folds $N_FOLDS"
else
   output_file="${base_filename}.kmeans.${LAT_WEIGHTS}.k${n_components}.n_init${N_INIT}.${REFERENCE}_gap.nc"
fi

if test "x$RESTRICT_TO_BASE_PERIOD" = "xyes" ; then
   opts="$opts --restrict-to-base-period"
fi

if test "x$VERBOSE" = "xyes" ; then
   opts="$opts --verbose"
fi

$PYTHON "$RUN_KMEANS" $opts "$input_file" "$output_file"
