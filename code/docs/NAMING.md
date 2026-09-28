# 命名约定

## 当前人脸模块命名

项目已经从 RetinaFace + FaceNet 演进到 SCRFD + MobileFace/ArcFace，人脸检测/识别文件按当前职责重命名：

| 当前名字 | 说明 |
| --- | --- |
| `face_models.h` | 人脸检测/识别公共结构、常量和公开接口 |
| `face_detector.cc` | SCRFD 人脸检测后处理，以及检测/识别两个 RKNN runtime 的初始化 |
| `face_recognizer.cc` | MobileFace/ArcFace 512D 特征、人脸库、匹配、录入 |
| `run_face_detector()` | 输出人脸框和 5 点 |
| `extract_face_embedding()` | 输出 512D L2 归一化特征 |

`retinaface_facenet.h`、`retinaface.cc`、`facenet.cc` 这些历史文件名已经移除。

## 暂不改名的内容

- 主程序可执行名 `retinaface_facenet_spidev` 暂时保留，避免影响板端脚本和部署路径。
- `yolov5.cc` 和 `postprocess.cc` 暂时保留，后续可单独迁移为 `person_detector_yolo.cc` 和 `person_postprocess.cc`。

## 新增代码命名规则

- 线程/异步模块使用 `*Worker`，例如 `CameraFrameWorker`、`LcdRefreshWorker`。
- 只做模型封装的模块使用 `*_model` 或 `*_detector`。
- 只做业务状态的结构体使用 `*Runtime` 或 `*State`。
- 上位机协议相关命名保留 `employee_*` 前缀。

## 不建议的改动

- 不要在性能优化中继续扩大文件名修改范围。
- 不要同时改模型名、阈值和后处理函数名。
- 不要改动板端启动脚本引用的可执行文件名，除非同步提供兼容入口。
