#!/bin/bash
set -e

ROOT_PWD=$(cd "$(dirname "$0")" && pwd)

if [ "$1" = "clean" ]; then
	if [ -d "${ROOT_PWD}/build" ]; then
		rm -rf "${ROOT_PWD}/build"
		echo " ${ROOT_PWD}/build has been deleted!"
	fi

	if [ -d "${ROOT_PWD}/install" ]; then
		rm -rf "${ROOT_PWD}/install"
		echo " ${ROOT_PWD}/install has been deleted!"
	fi

	exit
fi

: "${LUCKFOX_SDK_PATH:=/home/hkh/luckfox/luckfox-pico}"
export LUCKFOX_SDK_PATH
libc_type="uclibc"
opt="retinaface_facenet_spidev"
DEVICE="LUCKFOX_PICO_PRO_MAX"

echo "Using pre-set configuration:"
echo "Libc: $libc_type | Project: $opt | Device: $DEVICE"

src_dir="${ROOT_PWD}/code"

if [[ -d "$src_dir" ]]; then
    if [ -d "${ROOT_PWD}/build" ]; then
        rm -rf "${ROOT_PWD}/build"
    fi
    mkdir -p "${ROOT_PWD}/build"
    cd "${ROOT_PWD}/build"

    if [ -z "$DEVICE" ]; then
        cmake .. -DEXAMPLE_DIR="$src_dir" -DEXAMPLE_NAME="$opt" -DLIBC_TYPE="$libc_type"
    else
        cmake .. -DEXAMPLE_DIR="$src_dir" -DEXAMPLE_NAME="$opt" -DLIBC_TYPE="$libc_type" -D"$DEVICE"=ON
    fi
    cmake --build . --target install -j
else
    echo "错误：目录 $src_dir 不存在！"
    echo "Error: Directory $src_dir does not exist!"
    exit 1
fi
