#!/bin/bash

while getopts ":i:" opt; do
  case $opt in
    i) ID="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

# zip on server and pull
if [ -z ${ID} ]; then
  echo $ID
  echo "NO ID"
  exit
  ssh daint 'zip -r ~/storage.zip /scratch/snx3000/bp000299/dexterous-robot-hand/storage/ -x $SCRATCH/dexterous-robot-hand/storage/experience/*'
  scp daint:~/storage.zip .
else
  echo $ID
  echo "GOT ID"
  exit
  ssh daint 'zip -r ~/storage.zip /scratch/snx3000/bp000299/dexterous-robot-hand/storage/experiments/$ID /scratch/snx3000/bp000299/dexterous-robot-hand/storage/saved_models/$ID'
  scp daint:~/storage.zip .
fi

# unzip locally and move to correct directory
unzip -n storage.zip -d storage/
cp -r storage/scratch/snx3000/bp000299/dexterous-robot-hand/storage/* storage/

# cleanup
rm -r storage/scratch
rm storage.zip