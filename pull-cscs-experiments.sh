#!/bin/bash

REMOVE=false
ONLY_STATS=false

while getopts "i:ds" opt; do
  case $opt in
    i) ID="$OPTARG"
    ;;
    d) REMOVE=true
    ;;
    s) ONLY_STATS=true
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

if [ $REMOVE ]; then
  echo "Deleting existing storage zip file."
  ssh daint "rm ~/storage.zip"
fi

# zip on server and pull
if [ -z ${ID} ]; then
  echo "Zipping entire storage."
  ZIP_COMMAND="zip -u -r ~/storage.zip /scratch/snx3000/bp000299/dexterous-robot-hand/storage/ -x /scratch/snx3000/bp000299/dexterous-robot-hand/storage/experience/\* "
  if [ $ONLY_STATS ]; then
    ZIP_COMMAND="$ZIP_COMMAND -x /scratch/snx3000/bp000299/dexterous-robot-hand/storage/saved_models/\*"
  fi
  echo $ZIP_COMMAND
else
  echo "Updating/Adding $ID to zip file."
  ZIP_COMMAND="zip -u -r ~/storage.zip /scratch/snx3000/bp000299/dexterous-robot-hand/storage/experiments/$ID/ "
  if [ ! $ONLY_STATS ]; then
    ZIP_COMMAND+="/scratch/snx3000/bp000299/dexterous-robot-hand/storage/saved_models/$ID/"
  fi
fi

# execute and download
ssh daint $ZIP_COMMAND
scp daint:~/storage.zip .

# unzip locally and move to correct directory
unzip -n storage.zip -d storage/
cp -r storage/scratch/snx3000/bp000299/dexterous-robot-hand/storage/* storage/

# cleanup
rm -r storage/scratch
rm storage.zip