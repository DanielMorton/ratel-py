#!/usr/bin/env bash

export DOCKER_IMAGE_NAME="ratel:test"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

docker system prune -f

docker build ${DIR}/.. -f ${DIR}/Dockerfile -t ${DOCKER_IMAGE_NAME}