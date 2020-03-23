#!/bin/sh

BASE_DIR=$(dirname $0)
BIN_DIR=$(cd "$BASE_DIR"; pwd)
DATA_DIR="${BIN_DIR}/../data"
LOGS_DIR="${BIN_DIR}/../logs"

LOG_FILE="${LOGS_DIR}/$(basename -s "sh" $0)log"

WGET="wget"
EXTRACT="gunzip"

HADOBS_URL="https://www.metoffice.gov.uk/hadobs"
DATASET_NAME="hadisst"

DATAFILES="HadISST_sst.nc.gz HadISST_ice.nc.gz"

#_______________________________________________
timestamp() {
   date -u "+%Y-%m-%dT%H:%M:%s%z"
}

#_______________________________________________
create_datafile_url() {
   _url="${HADOBS_URL}/${DATASET_NAME}/data/$1"
}

#_______________________________________________
fetch_datafile() {
   $WGET -a "$LOG_FILE" -O "$2" $1
}

if test -f "$LOG_FILE" ; then
   rm "$LOG_FILE"
fi

if test ! -d "$LOGS_DIR" ; then
   mkdir "$LOGS_DIR"
   echo "Created log directory: $LOGS_DIR" > "$LOG_FILE"
fi

if test ! -d "$DATA_DIR" ; then
   echo "Creating data directory: $DATA_DIR" >> "$LOG_FILE"
   mkdir "$DATA_DIR"
fi

for f in $DATAFILES ; do

   echo "Fetching datafile: $f" >> "$LOG_FILE"

   dest="${DATA_DIR}/$f"
   timestamp_file="${DATA_DIR}/$f.timestamp"

   create_datafile_url $f
   src_url="$_url"

   echo "Fetching from: $src_url" >> "$LOG_FILE"
   fetch_datafile "$src_url" "$dest"

   status="$?"
   if test "x$status" != "x0" ; then
      msg="Error: failed to fetch datafile (exit code $status)"
      echo "$msg" >> "$LOG_FILE"
      echo "$msg"
      exit "$status"
   fi

   echo "Writing timestamp file: $timestamp_file" >> "$LOG_FILE"
   timestamp > "$timestamp_file"

   echo "Extracting datafile: $dest" >> "$LOG_FILE"
   $EXTRACT "$dest"

done
