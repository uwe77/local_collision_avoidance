#!/usr/bin/env bash

REPOSITORY="argnctu/robotx2022"
TAG="ipc-18.04"

IMG="${REPOSITORY}:${TAG}"

docker pull "${IMG}"
