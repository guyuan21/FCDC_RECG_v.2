import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['onnx', 'rknn'], default='onnx')
    parser.add_argument('--model', default='./det_25g.onnx')
    parser.add_argument('--dataset', default='../../dataset/retinaface_dataset.txt')
    parser.add_argument('--image', default='../retinaface/entire13.jpg')
    parser.add_argument('--image-dir', default=None)
    parser.add_argument('--out-dir', default='./scrfd_result')
    parser.add_argument('--input-size', default='640,640',
                        help='width,height, for example 640,640')
    parser.add_argument('--score-thresh', type=float, default=0.5)
    parser.add_argument('--nms-thresh', type=float, default=0.4)
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--preview-width', type=int, default=960)
    parser.add_argument('--no-save', action='store_true')
    return parser.parse_args()


def parse_input_size(value):
    width, height = value.split(',')
    return int(width), int(height)


def collect_images(args):
    if args.image_dir:
        image_dir = Path(args.image_dir)
        images = []
        for pattern in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
            images.extend(image_dir.glob(pattern))
        images = sorted(images)
        if args.max_images > 0:
            images = images[:args.max_images]
        return images
    return [Path(args.image)]


def preprocess(img, input_size, backend):
    input_w, input_h = input_size
    img_h, img_w = img.shape[:2]
    img_ratio = img_h / img_w
    model_ratio = input_h / input_w
    if img_ratio > model_ratio:
        new_h = input_h
        new_w = int(new_h / img_ratio)
    else:
        new_w = input_w
        new_h = int(new_w * img_ratio)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    det_img = np.zeros((input_h, input_w, 3), dtype=np.uint8)
    det_img[:new_h, :new_w, :] = resized
    det_scale = new_h / img_h

    # InsightFace SCRFD uses RGB input with mean=127.5 and std=128.0.
    rgb = det_img[..., ::-1]
    if backend == 'onnx':
        infer_img = (rgb.astype(np.float32) - 127.5) / 128.0
        infer_img = np.transpose(infer_img, (2, 0, 1))
        infer_img = np.expand_dims(infer_img, 0)
    else:
        infer_img = np.expand_dims(rgb, 0)
    return infer_img, det_scale


def make_anchor_centers(input_size, stride, num_anchors=2):
    input_w, input_h = input_size
    feat_h = input_h // stride
    feat_w = input_w // stride
    anchor_centers = np.stack(np.mgrid[:feat_h, :feat_w][::-1], axis=-1)
    anchor_centers = (anchor_centers.astype(np.float32) * stride).reshape((-1, 2))
    if num_anchors > 1:
        anchor_centers = np.stack([anchor_centers] * num_anchors, axis=1).reshape((-1, 2))
    return anchor_centers


def distance2bbox(points, distance):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance):
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, 0] + distance[:, i]
        py = points[:, 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def nms(dets, thresh):
    if dets.shape[0] == 0:
        return []
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
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
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def decode_outputs(outputs, input_size, det_scale, score_thresh, nms_thresh, img_shape):
    strides = [8, 16, 32]
    score_outputs = outputs[:3]
    bbox_outputs = outputs[3:6]
    kps_outputs = outputs[6:9]

    boxes_all = []
    scores_all = []
    kps_all = []
    for score, bbox_pred, kps_pred, stride in zip(score_outputs, bbox_outputs, kps_outputs, strides):
        score = np.asarray(score).reshape(-1)
        bbox_pred = np.asarray(bbox_pred).reshape(-1, 4)
        kps_pred = np.asarray(kps_pred).reshape(-1, 10)

        anchor_centers = make_anchor_centers(input_size, stride)
        count = min(score.shape[0], bbox_pred.shape[0], kps_pred.shape[0], anchor_centers.shape[0])
        score = score[:count]
        bbox_pred = bbox_pred[:count]
        kps_pred = kps_pred[:count]
        anchor_centers = anchor_centers[:count]

        pos_inds = np.where(score >= score_thresh)[0]
        if pos_inds.size == 0:
            continue

        bboxes = distance2bbox(anchor_centers, bbox_pred * stride)
        kps = distance2kps(anchor_centers, kps_pred * stride)
        boxes_all.append(bboxes[pos_inds])
        scores_all.append(score[pos_inds])
        kps_all.append(kps[pos_inds])

    if not boxes_all:
        return np.empty((0, 15), dtype=np.float32)

    boxes = np.vstack(boxes_all) / det_scale
    scores = np.concatenate(scores_all)
    kps = np.vstack(kps_all) / det_scale

    img_h, img_w = img_shape[:2]
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, img_w)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, img_h)
    kps[:, 0::2] = np.clip(kps[:, 0::2], 0, img_w)
    kps[:, 1::2] = np.clip(kps[:, 1::2], 0, img_h)

    order = scores.argsort()[::-1]
    boxes = boxes[order]
    scores = scores[order]
    kps = kps[order]

    dets = np.hstack((boxes, scores[:, None])).astype(np.float32)
    keep = nms(dets, nms_thresh)
    dets = dets[keep]
    kps = kps[keep]
    return np.hstack((dets, kps)).astype(np.float32)


def draw_label(img, text, x, y, font_scale, thickness):
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                         font_scale, thickness)
    x = max(0, min(x, img.shape[1] - tw - 1))
    y = max(th + baseline + 4, min(y, img.shape[0] - 1))
    cv2.rectangle(img, (x, y - th - baseline - 4), (x + tw + 6, y + baseline + 2),
                  (0, 0, 0), -1)
    cv2.putText(img, text, (x + 3, y - 2), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_result(img, dets, args, img_path):
    short_side = min(img.shape[:2])
    box_thickness = max(2, short_side // 800)
    point_radius = max(4, short_side // 350)
    font_scale = max(0.7, short_side / 2200.0)
    text_thickness = max(1, box_thickness // 2)
    colors = [(0, 0, 255), (0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 0, 0)]

    for det in dets:
        data = list(map(int, det))
        cv2.rectangle(img, (data[0], data[1]), (data[2], data[3]),
                      (0, 0, 255), box_thickness)
        draw_label(img, '{:.4f}'.format(float(det[4])), data[0], data[1] - 6,
                   font_scale, text_thickness)
        points = [(data[5], data[6]), (data[7], data[8]), (data[9], data[10]),
                  (data[11], data[12]), (data[13], data[14])]
        for pt, color in zip(points, colors):
            cv2.circle(img, pt, point_radius, color, -1, cv2.LINE_AA)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(img_path).stem
    result_path = out_dir / (stem + '_result1.jpg')
    preview_path = out_dir / (stem + '_preview1.jpg')
    cv2.imwrite(str(result_path), img)

    if args.preview_width > 0 and img.shape[1] > args.preview_width:
        preview_h = int(img.shape[0] * args.preview_width / img.shape[1])
        preview = cv2.resize(img, (args.preview_width, preview_h), interpolation=cv2.INTER_AREA)
    else:
        preview = img
    cv2.imwrite(str(preview_path), preview)
    print('save image in', result_path)
    print('save preview in', preview_path)


def run_image(runner, img_path, args, input_size, input_name=None):
    img = cv2.imread(str(img_path))
    if img is None:
        print('RESULT image={} error=read_failed'.format(img_path))
        return np.empty((0, 15), dtype=np.float32)

    infer_img, det_scale = preprocess(img, input_size, args.backend)
    if args.backend == 'onnx':
        outputs = runner.run(None, {input_name: infer_img})
    else:
        outputs = runner.inference(inputs=[infer_img], data_format='nhwc')
    dets = decode_outputs(outputs, input_size, det_scale, args.score_thresh,
                          args.nms_thresh, img.shape)

    best_score = float(dets[0, 4]) if len(dets) else 0.0
    print('RESULT image={} faces={} best_score={:.6f}'.format(
        Path(img_path).name, len(dets), best_score))
    if len(dets) > 0:
        d = dets[0]
        points = [(int(d[5]), int(d[6])), (int(d[7]), int(d[8])), (int(d[9]), int(d[10])),
                  (int(d[11]), int(d[12])), (int(d[13]), int(d[14]))]
        print('best_box=({} {} {} {}) landmarks={}'.format(
            int(d[0]), int(d[1]), int(d[2]), int(d[3]), points))

    if not args.no_save:
        draw_result(img, dets, args, img_path)
    return dets


if __name__ == '__main__':
    args = parse_args()
    input_size = parse_input_size(args.input_size)

    input_name = None
    if args.backend == 'onnx':
        print('--> Loading ONNX model')
        runner = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
        input_name = runner.get_inputs()[0].name
        print('input:', input_name, runner.get_inputs()[0].shape)
        print('outputs:', [(o.name, o.shape) for o in runner.get_outputs()])
        print('done')
    else:
        from rknn.api import RKNN
        runner = RKNN()
        print('--> Config model')
        runner.config(mean_values=[[127.5, 127.5, 127.5]],
                      std_values=[[128.0, 128.0, 128.0]],
                      target_platform='rv1106')
        print('done')

        print('--> Loading model')
        ret = runner.load_onnx(model=args.model,
                               inputs=['input.1'],
                               input_size_list=[[1, 3, input_size[1], input_size[0]]])
        if ret != 0:
            print('Load model failed!')
            exit(ret)
        print('done')

        print('--> Building model')
        ret = runner.build(do_quantization=True, dataset=args.dataset)
        if ret != 0:
            print('Build model failed!')
            exit(ret)
        print('done')

        print('--> Init runtime environment')
        ret = runner.init_runtime()
        if ret != 0:
            print('Init runtime environment failed!')
            exit(ret)
        print('done')

    image_paths = collect_images(args)
    face_images = 0
    total_faces = 0
    best_scores = []
    for image_path in image_paths:
        dets = run_image(runner, image_path, args, input_size, input_name)
        total_faces += len(dets)
        if len(dets):
            face_images += 1
            best_scores.append(float(dets[0, 4]))

    avg_best = float(np.mean(best_scores)) if best_scores else 0.0
    min_best = float(np.min(best_scores)) if best_scores else 0.0
    max_best = float(np.max(best_scores)) if best_scores else 0.0
    print('SUMMARY model={} images={} images_with_face={} total_faces={} avg_best={:.6f} min_best={:.6f} max_best={:.6f}'.format(
        args.model, len(image_paths), face_images, total_faces, avg_best, min_best, max_best))
    if args.backend == 'rknn':
        runner.release()
