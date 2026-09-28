# 性能与帧率策略

## 当前线程模型

主程序采用保守的三层解耦：

1. 主线程：状态机、人脸检测/识别、在岗检测调度、协议处理。
2. 摄像头线程：`CameraFrameWorker` 持续读取摄像头，只保留最新帧。
3. LCD 线程：`LcdRefreshWorker` 负责 BGR565 转换和 SPI 同步传输。

这样可以避免两个慢 I/O 操作拖住主循环：

- `cv::VideoCapture >> frame`
- `LCD_Display()`

## 推理降频

以下参数控制推理频率，默认值偏保守：

- `FACE_DETECT_INTERVAL_MS = 250`：人脸检测约 4Hz。
- `FACE_RECOG_INTERVAL_MS = 150`：识别特征提取最短间隔。
- `DUTY_INFER_INTERVAL_MS = 800`：在岗检测约 1.25Hz。
- `LCD_FLUSH_INTERVAL_MS = 25`：LCD 提交上限约 40Hz，实际受 SPI 限制。
- 空闲状态默认只做人脸检测预览，不做人脸特征提取和库匹配；录入、签到、签退会话才启用识别。需要恢复旧式空闲持续识别时设置 `ATTENDANCE_IDLE_FACE_RECOG=1`。
- `ATTENDANCE_LCD_SPI_MHZ`：板端临时指定 LCD SPI 频率，默认 35MHz，例如 `ATTENDANCE_LCD_SPI_MHZ=30 ./retinaface_facenet_spidev ...` 可临时降回 30MHz。
- `ATTENDANCE_SPI_BUFSIZ`：启动脚本会在系统允许时把 `/sys/module/spidev/parameters/bufsiz` 调到默认 `65536`，减少一帧 LCD 被拆分写入的次数。

如果要提升流畅度，优先观察 CPU/NPU 余量和温度，再小步调整：

- 人脸检测可尝试 `200ms`，风险中等。
- 在岗检测不建议低于 `500ms`，否则会影响人脸识别响应。
- 240x320 RGB565 全屏每帧约 153.6KB。20MHz SPI 理论上限约 16fps，实际 13-14fps 属于正常范围；30MHz 理论上限约 24fps，但仍会受内核 SPI、分块写入和面板接收能力影响。
- LCD 启动日志会打印 `SPI speed request/actual` 和 `SPI write chunk`。如果实际速度没有高于 20MHz，优先检查设备树/内核 SPI 上限；如果 `SPI write chunk` 仍是约 `3840`，说明 `spidev.bufsiz` 仍是 4096，需要通过启动脚本、内核参数或设备树提高。
- 直接把 LCD 提交上限调得更高意义不大，SPI 同步传输仍是瓶颈。

## 低风险优化原则

- 不改模型输入尺寸和预处理。
- 不改识别阈值，除非已有本人/陌生人 `dist` 数据支撑。
- I/O 线程采用“只保留最新帧”的策略，允许丢帧，不积压队列。
- 主循环使用预分配 `cv::Mat`，避免热路径反复分配。
- 签退进入睡眠后关闭摄像头和 YOLO 模型，降低持续负载。

## 测试建议

板端测试时同时观察：

- 屏幕主循环 FPS
- LCD FPS
- 签到识别 `dist`
- 在岗 person 框是否稳定
- 串口命令响应延迟

如果摄像头线程启用后出现 ISP 异常，优先回看 `CameraFrameWorker` 的启动/停止时机，而不是修改模型逻辑。
