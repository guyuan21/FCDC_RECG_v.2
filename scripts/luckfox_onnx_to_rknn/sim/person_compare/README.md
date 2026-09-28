# SCRFD Person Compare

This folder provides a host-side test script for `scrfd_person_2.5g.onnx`.

The current project only has `yolov5n.rknn`, not a YOLO ONNX model. RKNN Toolkit2 cannot run an already exported RKNN model in PC simulator through `load_rknn`; it must run on target, or the original YOLO ONNX must be supplied. Therefore:

- SCRFD person ONNX can be tested directly on the host.
- YOLO comparison is supported when a YOLOv5 ONNX path is passed with `--yolo-onnx`.
- For true accuracy, provide YOLO-format labels with `--label-dir`. Without labels, the script produces visual comparison plus count, best score, and inference-time statistics.

## Run SCRFD Person

```bash
cd /home/hkh/luckfox/Fcdc_Luckfox_version2/scripts/luckfox_onnx_to_rknn/sim/person_compare

/home/hkh/miniconda3/envs/rknn/bin/python person_compare.py \
  --scrfd ../../model/scrfd_person_2.5g.onnx \
  --image ../../dataset/pic/yolov5/bus.jpg
```

## Run On A Folder

```bash
/home/hkh/miniconda3/envs/rknn/bin/python person_compare.py \
  --scrfd ../../model/scrfd_person_2.5g.onnx \
  --image-dir /path/to/person_test_images \
  --out-dir ./person_compare_result
```

## Optional YOLO ONNX Compare

```bash
/home/hkh/miniconda3/envs/rknn/bin/python person_compare.py \
  --scrfd ../../model/scrfd_person_2.5g.onnx \
  --yolo-onnx /path/to/yolov5n.onnx \
  --image-dir /path/to/person_test_images
```

## Optional Label Metrics

Labels are standard YOLO txt files:

```text
0 cx cy w h
```

Run:

```bash
/home/hkh/miniconda3/envs/rknn/bin/python person_compare.py \
  --scrfd ../../model/scrfd_person_2.5g.onnx \
  --image-dir /path/to/person_test_images \
  --label-dir /path/to/person_test_labels
```

Outputs:

- `person_compare_result/images/*_compare.jpg`
- `person_compare_result/summary.csv`
