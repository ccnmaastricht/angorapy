#!/bin/bash

REMOVE=false
ONLY_STATS=false
BASE_DIRECTORY="\$SCRATCH"

ARGS=""
while [ $# -gt 0 ]
do
  while getopts i:b:ds flag
  do
    case "${flag}" in
      i) ID="${OPTARG}";;
      b) BASE_DIRECTORY="${OPTARG}";;
      d) REMOVE=true;;
      s) ONLY_STATS=true;;
      \?) echo "Invalid option -$OPTARG" >&2;;
    esac
  done
  shift $((OPTIND-1))
  ARGS="${ARGS} $1 "
  shift
done

POSARGS=($ARGS)
HOST=${POSARGS[0]}
BASE_DIRECTORY=$(ssh $HOST echo $BASE_DIRECTORY)

echo "PULLING $ID"
echo "FROM $BASE_DIRECTORY"

if [ "$REMOVE" = true ]; then
  echo "Deleting existing storage zip file."
  ssh daint "rm ~/storage.zip"
fi

# zip on server and pull
if [ -z ${ID} ]; then
  echo "Zipping entire storage."
  ZIP_COMMAND="zip -or $BASE_DIRECTORY/storage.zip $BASE_DIRECTORY/ -x $BASE_DIRECTORY/experience/\* "
  if [ "$ONLY_STATS" = true ]; then
    ZIP_COMMAND="$ZIP_COMMAND -x $BASE_DIRECTORY/saved_models/\*"
  fi
else
  echo "Updating/Adding $ID to zip file."
  ZIP_COMMAND="zip -or $BASE_DIRECTORY/storage.zip $BASE_DIRECTORY/experiments/$ID/ "
  if [ "$ONLY_STATS" = false ]; then
    ZIP_COMMAND="$ZIP_COMMAND $BASE_DIRECTORY/saved_models/$ID/"
  fi
fi

echo $ZIP_COMMAND

# execute and download
ssh $HOST $ZIP_COMMAND
scp $HOST:$BASE_DIRECTORY/storage.zip .

unzip -o storage.zip -d storage/
cp -r storage/$BASE_DIRECTORY/* storage/

# cleanup
IFS="/" read -a PATH_ARRAY <<< $BASE_DIRECTORY
rm -r storage/$PATH_ARRAY[0]
rm storage.zip