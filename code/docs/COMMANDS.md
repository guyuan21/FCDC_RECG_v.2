# 命令行用法

本文档集中记录项目内需要在 shell 中执行的命令。

## 主机编译

在仓库根目录执行：

```bash
cmake --build build -j
cmake --install build
```

`cmake --build` 只更新 `build/retinaface_facenet_spidev`；运行脚本通常使用 `install/RV1106_demo/` 下的安装产物，所以改完代码后必须执行 `cmake --install build`。

全量构建：

```bash
./build.sh
```

## 板端启动

进入板端安装目录：

```bash
cd /root/scrfd_mobilefacenet_demo
chmod a+x retinaface_facenet_spidev run_retinaface_facenet_spidev.sh
./run_retinaface_facenet_spidev.sh
```

直接调用主程序：

```bash
./retinaface_facenet_spidev \
  ./model/facedet.rknn \
  ./model/facerecg.rknn \
  ./faces \
  ./model/yolov5n.rknn
```

## 清空测试库

程序停止后执行：

```bash
./run_retinaface_facenet_spidev.sh --reset-db
```

直接调用主程序：

```bash
./retinaface_facenet_spidev --reset-db ./faces
./retinaface_facenet_spidev --reset-db ./faces /dev/ttyS1 115200
```

只清人脸库和映射，不操作 AS608：

```bash
./retinaface_facenet_spidev --reset-db ./faces -
```

清库命令会删除 `faces/` 下员工目录、`.face_feature_cache/`、`fingerprint_map.txt`，并在未跳过指纹模块时调用 AS608 `empty` 清空模板库。

## 环境变量

常用配置：

```bash
export ATTENDANCE_CONTROL_DEV=/dev/ttyS2
export ATTENDANCE_UART_UPLOAD_DEV=/dev/ttyS4
export ATTENDANCE_FINGERPRINT_DEV=/dev/ttyS1
export ATTENDANCE_FINGERPRINT_BAUD=115200
export ATTENDANCE_FINGERPRINT_PAGE_NUM=300
export ATTENDANCE_YOLO_MODEL=./model/yolov5n.rknn
export ATTENDANCE_LCD_FORCE_2_8=1
export ATTENDANCE_LCD_SPI_MHZ=35
export ATTENDANCE_TIMEZONE=CST-8
export ATTENDANCE_IDLE_FACE_RECOG=0
```

`ATTENDANCE_TIMEZONE` 默认就是 `CST-8`，用于板端 LCD/本地日志显示北京时间；协议上传仍是 Unix timestamp 秒。

日志开关：

```bash
ATTENDANCE_PROTOCOL_LOG=1 ./run_retinaface_facenet_spidev.sh
ATTENDANCE_DB_VERBOSE=1 ./run_retinaface_facenet_spidev.sh
```

## 控制输入调试

控制输入只保留手动指纹调试。正式签到、签退和员工管理走 UART4 员工管理协议，业务状态不再通过控制输入切换。

```text
FP_ENROLL:<index_or_worker_id>
FP_DELETE:<index_or_worker_id>
```

## 时间校准

查看板端时间：

```bash
date '+LOCAL %F %T %z'
date -u '+UTC   %F %T'
date +%s
```

设置 UTC 时间后写入硬件时钟：

```bash
date -u -s '2026-06-15 07:41:50'
hwclock -w -u
```

## AS608 辅助测试

如果已单独编译 `code/tools/as608_test.cc`，可直接操作 AS608 页号：

```bash
./as608_test /dev/ttyS1 115200 info
./as608_test /dev/ttyS1 115200 enroll 1
./as608_test /dev/ttyS1 115200 search
./as608_test /dev/ttyS1 115200 delete 1
```
