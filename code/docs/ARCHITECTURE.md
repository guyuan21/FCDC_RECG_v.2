# 架构说明

## 主流程

`main.cc` 是业务状态机，核心状态如下：

- `DETECT_STATE_FACE_ACTIVE`：人脸检测和识别，用于签到、签退、协议录入。
- `DETECT_STATE_DUTY_ACTIVE`：在岗检测，用 YOLOv5 person 判断是否有人在工作区域。
- `DETECT_STATE_SLEEP`：签退后的低功耗状态，关闭摄像头和在岗模型，等待员工管理协议请求。

## 模块边界

- 摄像头采集：`CameraFrameWorker` 独立线程拥有 `cv::VideoCapture`，主循环只读取最新帧副本。
- LCD 显示：`LcdRefreshWorker` 异步执行 `LCD_Display()`，主循环提交最新 UI 帧，显示线程可丢旧帧。
- 人脸检测：`face_detector.cc` 内部实际为 SCRFD 500M 后处理，输出仍复用 `object_detect_result_list`。
- 人脸识别：`face_recognizer.cc` 使用 112x112 MobileFace/ArcFace 输入，输出 512D L2 归一化特征。
- 在岗检测：`yolov5.cc` + `postprocess.cc` 输出 person 框，主程序只用分数和面积判断在岗。
- 协议：`employee_management.cc` 和 `employee_protocol.cc` 只处理上位机命令和结果帧。

## 识别效果保护边界

以下内容属于识别效果敏感区，保守优化时不要随意改动：

- `FACE_RECOG_THRESHOLD`、`FACE_RECOG_MARGIN`
- SCRFD/ArcFace 输入尺寸、RGB/BGR 预处理、关键点对齐模板
- `match_face()` 的最近距离和 margin 判断
- 录入样本数量与多帧确认逻辑

性能优化应优先放在 I/O 解耦、显示降频、摄像头采集、日志和内存复用上。
