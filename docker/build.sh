#!/usr/bin/env bash

# Check args
if [ "$#" -ne 1 ]; then
  echo "usage: ./build.sh IMAGE_NAME"
  return 1
fi

# Get this script's path
pushd `dirname $0` > /dev/null
dockerPath=`pwd`
latticenetPath="$(dirname "$dockerPath")"
latticenetContainingFolder="$(dirname "$latticenetPath")"
popd > /dev/null
#--build-arg workspace=$SCRIPTPATH\

# echo ${SCRIPTPATH}

# Build the docker image
docker build\
  --build-arg user=$USER\
  --build-arg uid=$UID\
  --build-arg home="/workspace"\
  --build-arg workspace=${latticenetContainingFolder}\
  --build-arg shell=$SHELL\
  -t $1 \
  -f Dockerfile .
