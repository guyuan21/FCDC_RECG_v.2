# Code 目录说明

本目录是 RV1106 端考勤程序的实际业务代码。当前目标是保持识别效果不变，同时把性能相关路径拆清楚，降低主循环阻塞。

## 目录

- `src/main.cc`：主状态机，负责待机、识别、在岗检测、签入签退、UI 调度。
- `src/camera_frame_worker.cc`：异步摄像头采集线程，只保留最新一帧给主循环使用。
- `src/face_detector.cc`：SCRFD 人脸检测 RKNN 初始化和后处理。
- `src/face_recognizer.cc`：MobileFace/ArcFace 特征提取、人脸库加载、匹配、录入。
- `src/yolov5.cc` 与 `src/postprocess.cc`：YOLOv5 person 在岗检测。
- `src/attendance_service.cc`：考勤状态和重复动作防抖，不再本地保存 CSV。
- `src/employee_*.cc`：上位机员工管理协议。
- `src/uart_control.cc`：串口唤醒、签退、指纹管理命令解析。

指纹页号通过 `faces/fingerprint_map.txt` 持久化为 `employee_id fingerprint_page`，不再依赖人脸库数组顺序推导。

## 文档

- `docs/ARCHITECTURE.md`：运行状态机和模块边界。
- `docs/PERFORMANCE.md`：线程模型、帧率策略和低风险优化边界。
- `docs/NAMING.md`：当前命名约定和保守改名策略。

## 构建

```bash
cmake --build /tmp/fcdc_v2_build -j2
cmake --install /tmp/fcdc_v2_build
```

安装产物位于：

```text
/home/hkh/luckfox/Fcdc_Luckfox_version2/install/RV1106_demo
```
