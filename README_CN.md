# Luckfox RV1106 人脸/指纹考勤项目说明

本项目运行在 Luckfox Pico Pro Max / RV1106 板端，主程序为 `retinaface_facenet_spidev`。当前版本以“上位机员工管理协议”为主线：上位机通过 UART4 发起员工录入、签到、签退；板端负责人脸采集、AS608 指纹、人脸特征库、在岗检测和 2.8 寸 SPI LCD 状态显示。

当前工作目录：

```text
/home/hkh/luckfox/Fcdc_Luckfox_version2
```

## 当前功能

- SCRFD 人脸检测模型：`model/facedet.rknn`
- MobileFace/ArcFace 人脸特征模型：`model/facerecg.rknn`
- YOLOv5n person 在岗检测模型：`model/yolov5n.rknn`
- AS608 指纹模块：默认 `/dev/ttyS1`，`115200`
- UART4 员工管理协议：默认 `/dev/ttyS4`
- 控制/调试串口：默认 `/dev/ttyS2`
- 2.8 寸 ILI9341 SPI LCD：`240x320`
- 人脸库目录数据库：`faces/<姓名>_<工号>_<权限>/sample_01.jpg`
- 人脸特征缓存：`faces/.face_feature_cache/`

## 目录结构

```text
Fcdc_Luckfox_version2/
├── CMakeLists.txt
├── build.sh
├── run_retinaface_facenet_spidev.sh
├── README_CN.md
├── code/
│   ├── include/
│   │   ├── employee_protocol.h        # 上位机员工管理协议
│   │   ├── employee_management.h      # 协议通道和请求队列
│   │   ├── face_models.h              # 人脸库、阈值、业务常量
│   │   ├── uart_control.h             # UART2/调试控制命令
│   │   └── AS608.h
│   ├── src/
│   │   ├── main.cc                    # 主状态机、LCD、考勤流程
│   │   ├── face_recognizer.cc         # 人脸库加载、注册、匹配、特征缓存
│   │   ├── face_detector.cc
│   │   ├── employee_protocol.cc
│   │   ├── employee_management.cc
│   │   ├── uart_control.cc
│   │   └── AS608.cc
│   ├── model/
│   └── tools/
│       └── face_db.txt                # 当前项目的人脸库模板
├── lib/
├── include/
├── tools/
└── install/RV1106_demo/               # 安装输出目录
```

## 编译和安装

推荐直接使用已有 `build/`：

```bash
cd /home/hkh/luckfox/Fcdc_Luckfox_version2
cmake --build build -j
cmake --install build
```

`cmake --build build -j` 只更新 `build/retinaface_facenet_spidev`。运行脚本通常使用安装目录里的程序，所以改完代码后必须执行：

```bash
cmake --install build
```

也可以全量重建：

```bash
./build.sh
```

安装产物：

```text
install/RV1106_demo/
├── retinaface_facenet_spidev
├── run_retinaface_facenet_spidev.sh
├── model/
├── lib/librga.so
└── face_db.txt
```

## 板端运行

将 `install/RV1106_demo/` 拷贝到板端，例如：

```text
/root/scrfd_mobilefacenet_demo
```

启动：

```bash
cd /root/scrfd_mobilefacenet_demo
chmod a+x retinaface_facenet_spidev run_retinaface_facenet_spidev.sh
./run_retinaface_facenet_spidev.sh
```

启动脚本会优先使用当前目录下的 `retinaface_facenet_spidev`，并自动选择 `faces/` 作为人脸库目录。

完整命令行用法见 [`code/docs/COMMANDS.md`](code/docs/COMMANDS.md)。

## 运行参数

主程序格式：

```bash
./retinaface_facenet_spidev \
  <detector.rknn> \
  <recognition.rknn> \
  <faces_dir|face_db.txt> \
  [yolov5n.rknn]
```

当前板端示例：

```bash
./retinaface_facenet_spidev \
  ./model/facedet.rknn \
  ./model/facerecg.rknn \
  ./faces \
  ./model/yolov5n.rknn
```

## 人脸数据保存方式

当前推荐方式是“目录即数据库”。当第三个参数传入 `faces/` 目录时，程序只扫描员工目录，不再读取 `faces/face_db.txt`。这样删除员工目录后不会被旧清单残留污染。

目录格式：

```text
faces/
├── 何开洪_1001_1/
│   ├── sample_01.jpg
│   ├── sample_02.jpg
│   └── sample_03.jpg
├── 段宇_1002_1/
│   ├── sample_01.jpg
│   ├── sample_02.jpg
│   └── sample_03.jpg
└── 杨康_1003_1/
    ├── sample_01.jpg
    ├── sample_02.jpg
    └── sample_03.jpg
```

员工目录名：

```text
<姓名>_<工号>_<权限>
```

字段：

- `姓名`：员工姓名或标识，不要包含空格。
- `工号`：上位机协议里的 `employee_id`。
- `权限`：`1` 表示允许，`0` 表示禁用。

在线录入成功后，板端会自动创建员工目录并保存三张样本：

```text
faces/<姓名>_<工号>_<权限>/sample_01.jpg
faces/<姓名>_<工号>_<权限>/sample_02.jpg
faces/<姓名>_<工号>_<权限>/sample_03.jpg
```

指纹页号独立保存在：

```text
faces/fingerprint_map.txt
```

格式：

```text
<employee_id> <fingerprint_page>
```

员工工号由上位机分配，只要求非 0 且唯一；AS608 的 `fingerprint_page` 由板端分配和维护。首次启动如果没有 `fingerprint_map.txt`，板端会按旧规则从当前人脸库顺序迁移一次，之后员工删除不会再改变其他员工的指纹页号。

### face_db.txt 的定位

`faces/face_db.txt` 只保留为旧文件模式兼容。只要程序参数传的是 `./faces` 目录，就会忽略 `faces/face_db.txt`，启动日志会出现：

```text
[DB] Directory mode: ignore legacy manifest: ./faces/face_db.txt
```

只有直接把第三个参数传成某个 `face_db.txt` 文件时，才会按旧格式读取：

```text
<name> <worker_id> <permission> <img_or_dir_1> [img_or_dir_2 ...]
```

当前模板见：

```text
code/tools/face_db.txt
```

### 特征缓存

程序会在 `faces/.face_feature_cache/` 下缓存每张图片的 512 维特征。图片路径、文件大小或修改时间变化后，缓存自动失效。

首次加载图片会执行：

```text
图片解码 -> 人脸检测/对齐 -> 特征提取 -> 写缓存
```

后续加载会直接读取缓存，启动速度会明显快于首次加载。

## 员工管理协议 UART4

串口参数：

```text
设备: /dev/ttyS4
波特率: 115200
数据位: 8
停止位: 1
校验: none
流控: none
```

帧格式：

```text
AA 55 | CMD | SEQ | LEN | PAYLOAD | CRC16_H | CRC16_L | FF
```

CRC：

```text
CRC16 Modbus，初值 0xFFFF，计算范围为 CMD + SEQ + LEN + PAYLOAD
CRC 以大端写入：高字节在前，低字节在后
```

所有多字节数字字段使用大端。

### 命令

```text
0x01 ENROLL_REQ       上位机请求录入员工
0x02 ENROLL_RSP       板端返回录入结果
0x10 CHECKIN_REQ      上位机请求签到
0x11 CHECKIN_OK       板端返回签到成功
0x12 CHECKIN_FAIL     板端返回签到失败
0x20 CHECKOUT_REQ     上位机请求签退
0x21 CHECKOUT_RSP     板端返回签退成功
0xFF ERROR            错误帧
```

### ENROLL_REQ 0x01

上位机发送，payload 长度 `5..36`：

```text
byte0-3   employee_id，uint32 big-endian
byte4..N  name，ASCII/UTF-8 字节
```

板端行为：

1. 启动人脸采集。
2. 采集 3 张人脸样本。
3. 录入 AS608 指纹。
4. 保存到 `faces/<姓名>_<工号>_<权限>/sample_XX.jpg`。
5. 返回 `ENROLL_RSP`。

### ENROLL_RSP 0x02

板端返回，payload 长度 `5`：

```text
byte0-3 employee_id
byte4   status
```

`status`：

```text
0x00 成功
0x01 人脸采集失败
0x02 指纹录入失败
0x03 employee_id 已存在
0x04 存储空间不足
```

### CHECKIN_REQ 0x10

上位机发送，payload 长度 `1`：

```text
byte0 固定 0x00
```

板端收到后进入签到识别。当前不做本地时间间隔限制。仍保留基本状态约束：已有在岗员工时不能再次签到。

### CHECKIN_OK 0x11

板端返回，payload 长度 `8`：

```text
byte0-3 employee_id
byte4-7 checkin_time，Unix timestamp 秒
```

### CHECKIN_FAIL 0x12

板端返回，payload 长度 `1`：

```text
0x01 人脸超时
0x02 指纹超时
0x03 人脸与指纹不匹配
0x04 未注册人员
0x05 设备忙
0x06 已有员工在岗
```

### CHECKOUT_REQ 0x20

上位机发送，payload 长度 `4`：

```text
byte0-3 employee_id
```

### CHECKOUT_RSP 0x21

板端返回，payload 长度 `16`：

```text
byte0-3   employee_id
byte4-7   checkin_time，Unix timestamp 秒
byte8-11  checkout_time，Unix timestamp 秒
byte12-15 duration_sec，checkout_time - checkin_time
```

示例，员工 `1001` 签退，在岗 `3600` 秒：

```text
AA 55 21 00 10 00 00 03 E9 66 66 99 80 66 66 A7 90 00 00 0E 10 46 7C FF
```

### ERROR 0xFF

payload 长度 `2`：

```text
byte0 orig_cmd
byte1 err_code
```

`err_code`：

```text
0x01 帧错误
0x02 CRC 错误
0x03 未知命令
0x04 长度错误
0x05 员工不存在
0x06 员工未签到，无法签退
0x07 超时
```

## 控制输入调试命令

控制输入默认 `/dev/ttyS2`。前台运行且标准输入是 TTY 时，可用键盘输入作为 fallback。

当前只保留手动指纹调试命令：

```text
FP_ENROLL:<index_or_worker_id>
FP_DELETE:<index_or_worker_id>
```

当前主线对接建议使用 UART4 员工管理协议，不再混用旧 `0x10/0x11` 累加校验协议；旧 attend/duty 本地上报逻辑已移除。

## 业务流程

### 录入

1. 上位机发送 `ENROLL_REQ`。
2. 板端采集 3 张人脸。
3. 板端录入 AS608 指纹。
4. 板端保存员工目录和样本图。
5. 板端写入内存人脸库并返回 `ENROLL_RSP`。

### 签到

1. 上位机发送 `CHECKIN_REQ`。
2. 板端进入人脸或指纹识别。
3. 识别成功后记录 `checkin_time = time(nullptr)`。
4. 板端返回 `CHECKIN_OK`。
5. 板端进入在岗检测状态。

### 签退

1. 上位机发送 `CHECKOUT_REQ(employee_id)`。
2. 板端校验该员工是否当前在岗。
3. 识别确认后记录 `checkout_time = time(nullptr)`。
4. 板端返回 `CHECKOUT_RSP`，包含签到时间、签退时间和在岗时长。
5. 板端清空当前在岗状态，回到待机/签到侧。

## 时间戳和时区

协议上传的是标准 Unix timestamp 秒，代码不主动加 8 小时，也不上传格式化后的北京时间。

如果上位机看到时间戳比实际北京时间多 8 小时，通常是板端系统时间把北京时间写进了 UTC 内核时钟。板端检查：

```bash
date '+LOCAL %F %T %z'
date -u '+UTC   %F %T'
date +%s
```

正确做法是校准板端 UTC 时间，上位机按本地时区显示。

默认运行只输出关键事件和错误。需要排查协议或人脸库加载细节时可临时打开：

```bash
ATTENDANCE_PROTOCOL_LOG=1 ./run_retinaface_facenet_spidev.sh
ATTENDANCE_DB_VERBOSE=1 ./run_retinaface_facenet_spidev.sh
```

例如北京时间 `2026-06-15 15:41:50`，UTC 应设置为 `2026-06-15 07:41:50`：

```bash
date -u -s '2026-06-15 07:41:50'
hwclock -w -u
```

`ATTENDANCE_TIMEZONE` 只影响板端 LCD/CSV 的本地显示，不应拿来修正协议时间戳。程序和启动脚本默认使用 `CST-8`，也就是北京时间 `+0800`。

## 环境变量

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

说明：

- `ATTENDANCE_CONTROL_DEV`：控制输入串口，仅用于手动指纹调试，默认 `/dev/ttyS2`。
- `ATTENDANCE_UART_UPLOAD_DEV`：员工管理协议串口，默认 `/dev/ttyS4`。
- `ATTENDANCE_FINGERPRINT_DEV`：AS608 串口，默认 `/dev/ttyS1`。
- `ATTENDANCE_FINGERPRINT_BAUD`：AS608 波特率，默认 `115200`。
- `ATTENDANCE_FINGERPRINT_PAGE_NUM`：AS608 搜索容量，默认读取模块容量，失败时使用 `300`。
- `ATTENDANCE_YOLO_MODEL`：YOLOv5n 模型路径。
- `ATTENDANCE_LCD_FORCE_2_8`：强制 2.8 寸 LCD 初始化。
- `ATTENDANCE_LCD_SPI_MHZ`：LCD SPI 频率，默认 `35`。
- `ATTENDANCE_TIMEZONE`：本地显示时区，默认 `CST-8`。
- `ATTENDANCE_IDLE_FACE_RECOG`：空闲时是否做人脸特征识别，默认关闭；关闭后空闲只做人脸检测预览，不会持续上报陌生人事件。调试旧行为时设为 `1`。

## 调试清库

推荐在程序未运行时直接执行命令行清库：

```bash
./run_retinaface_facenet_spidev.sh --reset-db
```

也可以直接调用主程序，不加载模型、不打开摄像头，执行完立即退出：

```bash
./retinaface_facenet_spidev --reset-db ./faces
./retinaface_facenet_spidev --reset-db ./faces /dev/ttyS1 115200
```

如果只想清人脸库和映射，不操作 AS608，可把指纹设备参数写成 `-`：

```bash
./retinaface_facenet_spidev --reset-db ./faces -
```

该命令会清空当前 `faces` 人脸库目录、`.face_feature_cache`、`fingerprint_map.txt`，并调用 AS608 `empty` 清空指纹模板库。日志会显示 `face/fp/map` 三项结果；如果指纹模块未连接，人脸库仍会被清空，但命令会返回部分失败。

## AS608 测试

默认连接：

```text
AS608 TXD -> Luckfox UART1 RX
AS608 RXD -> Luckfox UART1 TX
AS608 GND -> Luckfox GND
AS608 VCC -> 按模块要求接 3.3V 或 5V
```

测试命令：

```bash
./as608_test /dev/ttyS1 115200 info
./as608_test /dev/ttyS1 115200 enroll 1
./as608_test /dev/ttyS1 115200 search
./as608_test /dev/ttyS1 115200 delete 1
```

`as608_test` 直接操作 AS608 页号；主程序按人脸库加载顺序映射页号，不会把大工号直接写进 AS608 页号。

## 常见问题

### faces 里只有 face_db.txt，为什么还是空库

当前目录模式下 `face_db.txt` 会被忽略。需要创建员工目录：

```text
faces/何开洪_1001_1/sample_01.jpg
```

或者通过上位机 `ENROLL_REQ` 录入，板端自动创建目录。

### 删除了图片，为什么旧员工还出现

请确认运行的是新二进制。新版目录模式不会读取 `faces/face_db.txt`，不会再被旧清单污染。更新后执行：

```bash
cmake --build build -j
cmake --install build
```

然后把 `install/RV1106_demo/retinaface_facenet_spidev` 更新到板端。

### 人脸库加载慢

首次加载每张图片需要解码、检测、对齐和特征提取。第二次启动会命中特征缓存：

```text
faces/.face_feature_cache/
```

日志里关注：

```text
[DB] Feature cache hits=... built=... missing=...
```

### 上传时间戳偏 8 小时

代码上传的是 `time(nullptr)`。若偏 8 小时，优先检查板端 UTC 时钟是否被设置成北京时间。

### 摄像头打不开

程序启动会执行：

```text
RkLunch-stop.sh
```

如果仍失败，检查摄像头排线、sensor/ISP 日志和是否有其他进程占用。

### LCD 无显示或刷新慢

检查：

```bash
ls /dev/spidev0.0
cat /sys/module/spidev/parameters/bufsiz
```

启动脚本会尝试把 spidev bufsiz 设置为 `65536`。如果日志仍显示 `4096`，说明系统参数不可写或驱动限制，LCD 单帧会被拆成更多块，刷新更慢。

## 工具文件

- `code/tools/face_db.txt`：当前项目的人脸库模板。
- `code/tools/as608_test.cc`：AS608 测试工具源码。

## 模型转换

RKNN/ONNX 转换资料在：

```text
scripts/luckfox_onnx_to_rknn/
```
