#!/usr/bin/env bash

REPOSITORY="argnctu/robotx2022"
TAG="ubuntu-20.04"

IMG="${REPOSITORY}:${TAG}"

docker pull "${IMG}"
