#!/bin/sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -x "$SCRIPT_DIR/retinaface_facenet_spidev" ]; then
  RUN_DIR="$SCRIPT_DIR"
elif [ -x "$SCRIPT_DIR/install/RV1106_demo/retinaface_facenet_spidev" ]; then
  RUN_DIR="$SCRIPT_DIR/install/RV1106_demo"
elif [ -x "$SCRIPT_DIR/install/retinaface_facenet_spidev" ]; then
  RUN_DIR="$SCRIPT_DIR/install"
elif [ -x "$SCRIPT_DIR/install/uclibc/retinaface_facenet_spidev_pro_max_demo/retinaface_facenet_spidev" ]; then
  RUN_DIR="$SCRIPT_DIR/install/uclibc/retinaface_facenet_spidev_pro_max_demo"
else
  echo "Cannot find retinaface_facenet_spidev" >&2
  exit 1
fi

APP="$RUN_DIR/retinaface_facenet_spidev"
RETINA_MODEL="$RUN_DIR/model/facedet.rknn"
FACENET_MODEL="$RUN_DIR/model/facerecg.rknn"
YOLO_MODEL="$RUN_DIR/model/yolov5n.rknn"

require_file() {
  if [ ! -f "$1" ]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

if [ -d "$RUN_DIR/faces" ]; then
  FACE_DB="$RUN_DIR/faces"
elif [ -f "$RUN_DIR/face_db.txt" ]; then
  FACE_DB="$RUN_DIR/face_db.txt"
elif [ -f "$SCRIPT_DIR/code/tools/face_db.txt" ]; then
  FACE_DB="$SCRIPT_DIR/code/tools/face_db.txt"
else
  echo "Cannot find faces directory or face_db.txt" >&2
  exit 1
fi

: "${ATTENDANCE_SPI_BUFSIZ:=65536}"
if [ -n "$ATTENDANCE_SPI_BUFSIZ" ] && [ -w /sys/module/spidev/parameters/bufsiz ]; then
  echo "$ATTENDANCE_SPI_BUFSIZ" > /sys/module/spidev/parameters/bufsiz 2>/dev/null || true
fi

: "${ATTENDANCE_FINGERPRINT_DEV:=/dev/ttyS1}"
: "${ATTENDANCE_FINGERPRINT_BAUD:=115200}"
: "${ATTENDANCE_CONTROL_DEV:=/dev/ttyS2}"
: "${ATTENDANCE_LCD_FORCE_2_8:=1}"
: "${ATTENDANCE_TIMEZONE:=CST-8}"
export ATTENDANCE_FINGERPRINT_DEV
export ATTENDANCE_FINGERPRINT_BAUD
export ATTENDANCE_CONTROL_DEV
export ATTENDANCE_LCD_FORCE_2_8
export ATTENDANCE_TIMEZONE

if [ "$1" = "--reset-db" ] || [ "$1" = "reset-db" ] || [ "$1" = "--reset-all" ]; then
  exec "$APP" \
    --reset-db \
    "$FACE_DB" \
    "$ATTENDANCE_FINGERPRINT_DEV" \
    "$ATTENDANCE_FINGERPRINT_BAUD"
fi

require_file "$RETINA_MODEL"
require_file "$FACENET_MODEL"
require_file "$YOLO_MODEL"

exec "$APP" \
  "$RETINA_MODEL" \
  "$FACENET_MODEL" \
  "$FACE_DB" \
  "$YOLO_MODEL"
