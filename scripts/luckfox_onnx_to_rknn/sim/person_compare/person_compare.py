#!/usr/bin/env python3
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def parse_args():
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    parser = argparse.ArgumentParser(
        description='Compare SCRFD person ONNX with an optional YOLOv5 ONNX on the same images.')
    parser.add_argument('--scrfd', default=str(root / 'model' / 'scrfd_person_2.5g.onnx'))
    parser.add_argument('--yolo-onnx', default=None,
                        help='Optional YOLOv5 ONNX. Current project only has yolov5n.rknn, which cannot run on PC simulator by load_rknn.')
    parser.add_argument('--image', default=str(root / 'dataset' / 'pic' / 'yolov5' / 'bus.jpg'))
    parser.add_argument('--image-dir', default=None)
    parser.add_argument('--list', default=None, help='Text file containing image paths.')
    parser.add_argument('--out-dir', default=str(here / 'person_compare_result'))
    parser.add_argument('--input-size', default='640,640', help='width,height')
    parser.add_argument('--scrfd-score', type=float, default=0.35)
    parser.add_argument('--yolo-score', type=float, default=0.25)
    parser.add_argument('--nms', type=float, default=0.50)
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--label-dir', default=None,
                        help='Optional YOLO-format label dir for simple TP/FP/FN at IoU threshold.')
    parser.add_argument('--iou-thresh', type=float, default=0.50)
    parser.add_argument('--preview-width', type=int, default=1280)
    parser.add_argument('--no-save-images', action='store_true')
    return parser.parse_args()


def parse_input_size(text):
    w, h = text.split(',')
    return int(w), int(h)


def collect_images(args):
    paths = []
    if args.list:
        base = Path(args.list).resolve().parent
        for line in Path(args.list).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            p = Path(line)
            paths.append(p if p.is_absolute() else base / p)
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        for ext in IMAGE_EXTS:
            paths.extend(image_dir.glob(f'*{ext}'))
            paths.extend(image_dir.glob(f'*{ext.upper()}'))
        paths = sorted(set(paths))
    else:
        paths = [Path(args.image)]

    paths = [p for p in paths if p.exists()]
    if args.max_images > 0:
        paths = paths[:args.max_images]
    return paths


def nms_xyxy(boxes, scores, thresh):
    if len(boxes) == 0:
        return []
    boxes = boxes.astype(np.float32)
    scores = scores.astype(np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1 + 1) * np.maximum(0.0, y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-6)
        order = order[np.where(iou <= thresh)[0] + 1]
    return keep


def distance2bbox(points, distance):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance):
    preds = []
    for i in range(0, distance.shape[1], 2):
        preds.append(points[:, 0] + distance[:, i])
        preds.append(points[:, 1] + distance[:, i + 1])
    return np.stack(preds, axis=-1)


def make_anchor_centers(input_size, stride, num_anchors):
    input_w, input_h = input_size
    feat_h = input_h // stride
    feat_w = input_w // stride
    centers = np.stack(np.mgrid[:feat_h, :feat_w][::-1], axis=-1)
    centers = (centers.astype(np.float32) * stride).reshape((-1, 2))
    if num_anchors > 1:
        centers = np.stack([centers] * num_anchors, axis=1).reshape((-1, 2))
    return centers


def scrfd_preprocess(img_bgr, input_size):
    input_w, input_h = input_size
    img_h, img_w = img_bgr.shape[:2]
    img_ratio = img_h / img_w
    model_ratio = input_h / input_w
    if img_ratio > model_ratio:
        new_h = input_h
        new_w = int(new_h / img_ratio)
    else:
        new_w = input_w
        new_h = int(new_w * img_ratio)

    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    det_img = np.zeros((input_h, input_w, 3), dtype=np.uint8)
    det_img[:new_h, :new_w] = resized
    det_scale = new_h / img_h

    rgb = det_img[:, :, ::-1].astype(np.float32)
    blob = (rgb - 127.5) / 128.0
    blob = np.transpose(blob, (2, 0, 1))[None, :, :, :]
    return blob, det_scale


def decode_scrfd_person(outputs, input_size, det_scale, img_shape, score_thresh, nms_thresh):
    branch_count = len(outputs) // 3
    if branch_count <= 0:
        return np.empty((0, 5), dtype=np.float32), np.empty((0, 10), dtype=np.float32)

    strides = [8, 16, 32, 64, 128][:branch_count]
    score_outputs = outputs[:branch_count]
    bbox_outputs = outputs[branch_count:branch_count * 2]
    kps_outputs = outputs[branch_count * 2:branch_count * 3]

    boxes_all = []
    scores_all = []
    kps_all = []
    input_w, input_h = input_size
    for score, bbox_pred, kps_pred, stride in zip(score_outputs, bbox_outputs, kps_outputs, strides):
        score = np.asarray(score).reshape(-1)
        bbox_pred = np.asarray(bbox_pred).reshape(-1, 4)
        kps_pred = np.asarray(kps_pred).reshape(-1, 10)

        feat_count = (input_h // stride) * (input_w // stride)
        num_anchors = max(1, score.shape[0] // max(1, feat_count))
        centers = make_anchor_centers(input_size, stride, num_anchors)

        count = min(score.shape[0], bbox_pred.shape[0], kps_pred.shape[0], centers.shape[0])
        score = score[:count]
        bbox_pred = bbox_pred[:count]
        kps_pred = kps_pred[:count]
        centers = centers[:count]

        keep = np.where(score >= score_thresh)[0]
        if keep.size == 0:
            continue

        boxes = distance2bbox(centers, bbox_pred * stride)[keep]
        kps = distance2kps(centers, kps_pred * stride)[keep]
        boxes_all.append(boxes)
        scores_all.append(score[keep])
        kps_all.append(kps)

    if not boxes_all:
        return np.empty((0, 5), dtype=np.float32), np.empty((0, 10), dtype=np.float32)

    boxes = np.vstack(boxes_all) / det_scale
    scores = np.concatenate(scores_all)
    kps = np.vstack(kps_all) / det_scale

    img_h, img_w = img_shape[:2]
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, img_w - 1)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, img_h - 1)
    kps[:, 0::2] = np.clip(kps[:, 0::2], 0, img_w - 1)
    kps[:, 1::2] = np.clip(kps[:, 1::2], 0, img_h - 1)

    order = scores.argsort()[::-1]
    boxes = boxes[order]
    scores = scores[order]
    kps = kps[order]
    keep = nms_xyxy(boxes, scores, nms_thresh)
    boxes = boxes[keep]
    scores = scores[keep]
    kps = kps[keep]
    return np.hstack([boxes, scores[:, None]]).astype(np.float32), kps.astype(np.float32)


class ScrfdPersonOnnx:
    def __init__(self, model_path, input_size, score_thresh, nms_thresh):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        print('SCRFD input:', self.input_name, self.session.get_inputs()[0].shape)
        print('SCRFD outputs:', [(o.name, o.shape) for o in self.session.get_outputs()])

    def detect(self, img_bgr):
        blob, scale = scrfd_preprocess(img_bgr, self.input_size)
        start = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: blob})
        infer_ms = (time.perf_counter() - start) * 1000.0
        boxes, kps = decode_scrfd_person(outputs, self.input_size, scale, img_bgr.shape,
                                         self.score_thresh, self.nms_thresh)
        return boxes, kps, infer_ms


def yolo_letterbox(img_bgr, input_size):
    input_w, input_h = input_size
    h, w = img_bgr.shape[:2]
    scale = min(input_w / w, input_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    pad_w = input_w - new_w
    pad_h = input_h - new_h
    left = int(round(pad_w / 2 - 0.1))
    top = int(round(pad_h / 2 - 0.1))

    canvas = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas[top:top + new_h, left:left + new_w] = resized
    rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
    blob = np.transpose(rgb, (2, 0, 1))[None, :, :, :]
    return blob, scale, left, top


def decode_yolov5_onnx(outputs, input_size, img_shape, scale, pad_left, pad_top,
                       score_thresh, nms_thresh):
    pred = outputs[0]
    if isinstance(pred, list):
        pred = pred[0]
    pred = np.asarray(pred)
    pred = np.squeeze(pred)
    if pred.ndim != 2 or pred.shape[-1] < 6:
        raise RuntimeError(f'Unsupported YOLO output shape: {pred.shape}')

    boxes = pred[:, :4].astype(np.float32)
    obj = pred[:, 4].astype(np.float32)
    class_scores = pred[:, 5:].astype(np.float32)
    person_scores = class_scores[:, 0] if class_scores.shape[1] > 1 else class_scores.reshape(-1)
    scores = obj * person_scores
    keep = scores >= score_thresh
    if not np.any(keep):
        return np.empty((0, 5), dtype=np.float32)

    boxes = boxes[keep]
    scores = scores[keep]
    xyxy = np.zeros_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_left) / scale
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_top) / scale
    img_h, img_w = img_shape[:2]
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, img_w - 1)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, img_h - 1)

    order = scores.argsort()[::-1]
    xyxy = xyxy[order]
    scores = scores[order]
    keep_idx = nms_xyxy(xyxy, scores, nms_thresh)
    return np.hstack([xyxy[keep_idx], scores[keep_idx, None]]).astype(np.float32)


class YoloV5Onnx:
    def __init__(self, model_path, input_size, score_thresh, nms_thresh):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        input_shape = self.session.get_inputs()[0].shape
        if len(input_shape) == 4 and isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
            self.input_size = (int(input_shape[3]), int(input_shape[2]))
        print('YOLO input:', self.input_name, input_shape, 'use_size=', self.input_size)
        print('YOLO outputs:', [(o.name, o.shape) for o in self.session.get_outputs()])

    def detect(self, img_bgr):
        blob, scale, left, top = yolo_letterbox(img_bgr, self.input_size)
        start = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: blob})
        infer_ms = (time.perf_counter() - start) * 1000.0
        boxes = decode_yolov5_onnx(outputs, self.input_size, img_bgr.shape,
                                   scale, left, top, self.score_thresh, self.nms_thresh)
        return boxes, None, infer_ms


def load_yolo_labels(label_dir, image_path, img_shape):
    if not label_dir:
        return np.empty((0, 4), dtype=np.float32)
    label_path = Path(label_dir) / (Path(image_path).stem + '.txt')
    if not label_path.exists():
        return np.empty((0, 4), dtype=np.float32)

    img_h, img_w = img_shape[:2]
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        if cls != 0:
            continue
        cx, cy, w, h = map(float, parts[1:5])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes.append([x1, y1, x2, y2])
    return np.asarray(boxes, dtype=np.float32)


def box_iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.maximum(0.0, inter_x2 - inter_x1) * np.maximum(0.0, inter_y2 - inter_y1)
    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-6)


def simple_metrics(pred_boxes5, gt_boxes, iou_thresh):
    pred_boxes = pred_boxes5[:, :4] if len(pred_boxes5) else np.empty((0, 4), dtype=np.float32)
    ious = box_iou(pred_boxes, gt_boxes)
    matched_gt = set()
    tp = 0
    for i in range(len(pred_boxes)):
        if ious.shape[1] == 0:
            continue
        j = int(np.argmax(ious[i]))
        if ious[i, j] >= iou_thresh and j not in matched_gt:
            tp += 1
            matched_gt.add(j)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


def draw_boxes(img, boxes, color, name, kps=None):
    out = img.copy()
    h, w = out.shape[:2]
    thickness = max(2, min(h, w) // 500)
    font_scale = max(0.55, min(h, w) / 1300.0)
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box[:4].astype(int)
        score = float(box[4])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        text = f'{name} {score:.2f}'
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        y = max(th + base + 4, y1)
        cv2.rectangle(out, (x1, y - th - base - 4), (x1 + tw + 6, y + base + 2), (0, 0, 0), -1)
        cv2.putText(out, text, (x1 + 3, y - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        if kps is not None and idx < len(kps):
            pts = kps[idx].reshape(-1, 2).astype(int)
            for pt in pts:
                cv2.circle(out, tuple(pt), max(2, thickness), color, -1, cv2.LINE_AA)
    return out


def save_visual(out_dir, image_path, img, scrfd_boxes, scrfd_kps, yolo_boxes, preview_width):
    left = draw_boxes(img, scrfd_boxes, (0, 0, 255), 'SCRFD', scrfd_kps)
    panels = [left]
    if yolo_boxes is not None:
        panels.append(draw_boxes(img, yolo_boxes, (0, 180, 0), 'YOLO'))
    canvas = np.hstack(panels)
    if preview_width > 0 and canvas.shape[1] > preview_width:
        new_h = int(canvas.shape[0] * preview_width / canvas.shape[1])
        canvas = cv2.resize(canvas, (preview_width, new_h), interpolation=cv2.INTER_AREA)
    image_dir = Path(out_dir) / 'images'
    image_dir.mkdir(parents=True, exist_ok=True)
    out_path = image_dir / (Path(image_path).stem + '_compare.jpg')
    cv2.imwrite(str(out_path), canvas)
    return out_path


def summarize_model(rows, model_name):
    rows = [r for r in rows if r['model'] == model_name]
    if not rows:
        return
    count_sum = sum(int(r['count']) for r in rows)
    best_scores = [float(r['best_score']) for r in rows if float(r['best_score']) > 0]
    avg_ms = sum(float(r['infer_ms']) for r in rows) / len(rows)
    avg_best = sum(best_scores) / len(best_scores) if best_scores else 0.0
    print(f'SUMMARY {model_name}: images={len(rows)} total_det={count_sum} '
          f'avg_det={count_sum / len(rows):.2f} avg_best={avg_best:.4f} avg_infer_ms={avg_ms:.2f}')


def main():
    args = parse_args()
    input_size = parse_input_size(args.input_size)
    images = collect_images(args)
    if not images:
        raise SystemExit('No images found. Use --image or --image-dir.')

    cv2.setNumThreads(1)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scrfd = ScrfdPersonOnnx(args.scrfd, input_size, args.scrfd_score, args.nms)
    yolo = None
    if args.yolo_onnx:
        yolo = YoloV5Onnx(args.yolo_onnx, input_size, args.yolo_score, args.nms)
    else:
        print('YOLO ONNX not provided: SCRFD will be tested alone.')
        print('Current project has yolov5n.rknn; RKNN Toolkit cannot load_rknn for PC simulator inference.')

    csv_rows = []
    metric_totals = {}
    for image_path in images:
        img = cv2.imread(str(image_path))
        if img is None:
            print('READ_FAILED', image_path)
            continue

        scrfd_boxes, scrfd_kps, scrfd_ms = scrfd.detect(img)
        yolo_boxes = None
        yolo_ms = 0.0
        if yolo is not None:
            yolo_boxes, _, yolo_ms = yolo.detect(img)

        gt = load_yolo_labels(args.label_dir, image_path, img.shape)
        for model_name, boxes, infer_ms in (
                ('scrfd_person_2.5g', scrfd_boxes, scrfd_ms),
                ('yolov5', yolo_boxes, yolo_ms)):
            if boxes is None:
                continue
            best = float(boxes[0, 4]) if len(boxes) else 0.0
            row = {
                'image': Path(image_path).name,
                'model': model_name,
                'count': len(boxes),
                'best_score': f'{best:.6f}',
                'infer_ms': f'{infer_ms:.3f}',
            }
            if args.label_dir:
                tp, fp, fn = simple_metrics(boxes, gt, args.iou_thresh)
                row.update({'tp': tp, 'fp': fp, 'fn': fn, 'gt': len(gt)})
                totals = metric_totals.setdefault(model_name, {'tp': 0, 'fp': 0, 'fn': 0})
                totals['tp'] += tp
                totals['fp'] += fp
                totals['fn'] += fn
            csv_rows.append(row)
            print(f'RESULT image={Path(image_path).name} model={model_name} '
                  f'count={len(boxes)} best={best:.4f} infer_ms={infer_ms:.2f}')

        if not args.no_save_images:
            out_path = save_visual(out_dir, image_path, img, scrfd_boxes, scrfd_kps,
                                   yolo_boxes, args.preview_width)
            print('SAVE', out_path)

    csv_path = out_dir / 'summary.csv'
    fieldnames = ['image', 'model', 'count', 'best_score', 'infer_ms']
    if args.label_dir:
        fieldnames += ['tp', 'fp', 'fn', 'gt']
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print('CSV', csv_path)

    summarize_model(csv_rows, 'scrfd_person_2.5g')
    summarize_model(csv_rows, 'yolov5')
    for model_name, totals in metric_totals.items():
        tp, fp, fn = totals['tp'], totals['fp'], totals['fn']
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        print(f'METRIC {model_name}: tp={tp} fp={fp} fn={fn} '
              f'precision={precision:.4f} recall={recall:.4f} iou={args.iou_thresh}')


if __name__ == '__main__':
    main()
