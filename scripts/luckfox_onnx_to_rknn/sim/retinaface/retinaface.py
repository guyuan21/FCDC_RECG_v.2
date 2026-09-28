import os
import sys
import urllib
import urllib.request
import time
import argparse
from pathlib import Path
import numpy as np
import cv2
from math import ceil
from itertools import product as product

from rknn.api import RKNN
DATASET_PATH = './dataset.txt'
DEFAULT_QUANT = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='./scrfd_2.5g_bnkps.onnx')
    parser.add_argument('--dataset', default=DATASET_PATH)
    parser.add_argument('--image', default='./result.jpg')
    parser.add_argument('--image-dir', default=None)
    parser.add_argument('--out-dir', default='.')
    parser.add_argument('--score-thresh', type=float, default=0.5)
    parser.add_argument('--nms-thresh', type=float, default=0.2)
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--preview-width', type=int, default=960)
    parser.add_argument('--no-save', action='store_true')
    return parser.parse_args()


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

def letterbox_resize(image, size, bg_color):
    """
    letterbox_resize the image according to the specified size
    :param image: input image, which can be a NumPy array or file path
    :param size: target size (width, height)
    :param bg_color: background filling data 
    :return: processed image
    """
    if isinstance(image, str):
        image = cv2.imread(image)

    target_width, target_height = size
    image_height, image_width, _ = image.shape

    # 计算调整后的图像尺寸
    aspect_ratio = min(target_width / image_width, target_height / image_height)
    new_width = int(image_width * aspect_ratio)
    new_height = int(image_height * aspect_ratio)

    # 使用 cv2.resize() 进行等比缩放
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # 创建新的画布并进行填充
    result_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * bg_color
    offset_x = (target_width - new_width) // 2
    offset_y = (target_height - new_height) // 2
    result_image[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = image
    return result_image, aspect_ratio, offset_x, offset_y

def PriorBox(image_size): #image_size Support (320,320) and (640,640)
    anchors = []
    min_sizes = [[16, 32], [64, 128], [256, 512]]
    steps = [8, 16, 32]
    feature_maps = [[ceil(image_size[0] / step), ceil(image_size[1] / step)] for step in steps]
    for k, f in enumerate(feature_maps):
        min_sizes_ = min_sizes[k]
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in min_sizes_:
                s_kx = min_size / image_size[1]
                s_ky = min_size / image_size[0]
                dense_cx = [x * steps[k] / image_size[1] for x in [j + 0.5]]
                dense_cy = [y * steps[k] / image_size[0] for y in [i + 0.5]]
                for cy, cx in product(dense_cy, dense_cx):
                    anchors += [cx, cy, s_kx, s_ky]
    output = np.array(anchors).reshape(-1, 4)
    print("image_size:",image_size," num_priors=",output.shape[0])
    return output


def box_decode(loc, priors):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    variances = [0.1, 0.2]
    boxes = np.concatenate((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), axis=1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landm(pre, priors):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    variances = [0.1, 0.2]
    landmarks = np.concatenate((
        priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:]
    ), axis=1)
    return landmarks


def nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

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


def face_scores(conf):
    conf = conf.astype(np.float32)
    # Some RKNN builds expose RetinaFace class outputs as logits instead of
    # probabilities. Convert them to a human-readable 0..1 confidence only
    # when the values are clearly outside probability range.
    if conf.max() > 1.0 or conf.min() < 0.0:
        exp_conf = np.exp(conf - np.max(conf, axis=1, keepdims=True))
        conf = exp_conf / np.sum(exp_conf, axis=1, keepdims=True)
    return conf[:, 1]


def draw_label(img, text, x, y, font_scale, thickness):
    baseline = 0
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX,
                                         font_scale, thickness)
    x = max(0, min(x, img.shape[1] - tw - 1))
    y = max(th + baseline + 4, min(y, img.shape[0] - 1))
    cv2.rectangle(img,
                  (x, y - th - baseline - 4),
                  (x + tw + 6, y + baseline + 2),
                  (0, 0, 0),
                  -1)
    cv2.putText(img, text, (x + 3, y - 2),
                cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255),
                thickness, cv2.LINE_AA)


def run_retinaface_image(rknn, img_path, args):
    img = cv2.imread(str(img_path))
    if img is None:
        print("RESULT image={} error=read_failed".format(img_path))
        return []

    img_height, img_width, _ = img.shape
    model_height, model_width = (640, 640)
    letterbox_img, aspect_ratio, offset_x, offset_y = letterbox_resize(
        img, (model_height, model_width), 114)
    infer_img = letterbox_img[..., ::-1]  # BGR2RGB
    infer_img = np.expand_dims(infer_img, 0)

    outputs = rknn.inference(inputs=[infer_img], data_format='nhwc')
    loc, conf, landmarks = outputs

    priors = PriorBox(image_size=(model_height, model_width))
    boxes = box_decode(loc.squeeze(0), priors)

    scale = np.array([model_width, model_height, model_width, model_height])
    boxes = boxes * scale // 1
    boxes[..., 0::2] = np.clip((boxes[..., 0::2] - offset_x) / aspect_ratio, 0, img_width)
    boxes[..., 1::2] = np.clip((boxes[..., 1::2] - offset_y) / aspect_ratio, 0, img_height)
    scores = face_scores(conf.squeeze(0))

    landmarks = decode_landm(landmarks.squeeze(0), priors)
    scale_landmarks = np.array([model_width, model_height, model_width, model_height,
                                model_width, model_height, model_width, model_height,
                                model_width, model_height])
    landmarks = landmarks * scale_landmarks // 1
    landmarks[..., 0::2] = np.clip((landmarks[..., 0::2] - offset_x) / aspect_ratio, 0, img_width)
    landmarks[..., 1::2] = np.clip((landmarks[..., 1::2] - offset_y) / aspect_ratio, 0, img_height)

    inds = np.where(scores > args.score_thresh)[0]
    boxes = boxes[inds]
    landmarks = landmarks[inds]
    scores = scores[inds]

    order = scores.argsort()[::-1]
    boxes = boxes[order]
    landmarks = landmarks[order]
    scores = scores[order]

    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    if dets.shape[0] > 0:
        keep = nms(dets, args.nms_thresh)
        dets = dets[keep, :]
        landmarks = landmarks[keep]
        dets = np.concatenate((dets, landmarks), axis=1)
    else:
        dets = np.empty((0, 15), dtype=np.float32)

    best_score = float(dets[0, 4]) if len(dets) else 0.0
    print("RESULT image={} faces={} best_score={:.6f}".format(
        Path(img_path).name, len(dets), best_score))

    if not args.no_save:
        draw_retinaface_result(img, dets, args, img_path)

    return dets


def draw_retinaface_result(img, dets, args, img_path):
    short_side = min(img.shape[:2])
    box_thickness = max(2, short_side // 800)
    point_radius = max(4, short_side // 350)
    font_scale = max(0.7, short_side / 2200.0)
    text_thickness = max(1, box_thickness // 2)

    for data in dets:
        if data[4] < args.score_thresh:
            continue
        print("face @ (%d %d %d %d) %f" % (data[0], data[1], data[2], data[3], data[4]))
        text = "{:.4f}".format(data[4])
        data = list(map(int, data))
        cv2.rectangle(img, (data[0], data[1]),
                      (data[2], data[3]), (0, 0, 255), box_thickness)
        draw_label(img, text, data[0], data[1] - 6, font_scale, text_thickness)

        points = [(data[5], data[6]), (data[7], data[8]), (data[9], data[10]),
                  (data[11], data[12]), (data[13], data[14])]
        colors = [(0, 0, 255), (0, 255, 255), (255, 0, 255),
                  (0, 255, 0), (255, 0, 0)]
        print("landmarks:", points)
        for pt, color in zip(points, colors):
            cv2.circle(img, pt, point_radius, color, -1, cv2.LINE_AA)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(img_path).stem
    img_path_out = out_dir / (stem + '_result.jpg')
    cv2.imwrite(str(img_path_out), img)
    print("save image in", img_path_out)

    preview_path = out_dir / (stem + '_preview.jpg')
    if args.preview_width > 0 and img.shape[1] > args.preview_width:
        preview_h = int(img.shape[0] * args.preview_width / img.shape[1])
        preview = cv2.resize(img, (args.preview_width, preview_h), interpolation=cv2.INTER_AREA)
    else:
        preview = img
    cv2.imwrite(str(preview_path), preview)
    print("save preview in", preview_path)


if __name__ == '__main__':
    args = parse_args()

    # 创建RKNN对象 
    rknn = RKNN()

    # 预处理设置
    print('--> Config model')
    rknn.config(mean_values=[[104, 117, 123]], std_values=[[1, 1, 1]], target_platform="rv1106",
                quantized_algorithm="normal", quant_img_RGB2BGR=True)  # mmse
    print('done')

    # 载入模型
    print('--> Loading model')
    ret = rknn.load_onnx(model=args.model)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # 创建模型
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset=args.dataset)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # 初始化运行时环境
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # 运行
    print('--> Running model')
    image_paths = collect_images(args)
    face_images = 0
    total_faces = 0
    best_scores = []
    for image_path in image_paths:
        dets = run_retinaface_image(rknn, image_path, args)
        total_faces += len(dets)
        if len(dets) > 0:
            face_images += 1
            best_scores.append(float(dets[0, 4]))

    avg_best = float(np.mean(best_scores)) if best_scores else 0.0
    min_best = float(np.min(best_scores)) if best_scores else 0.0
    max_best = float(np.max(best_scores)) if best_scores else 0.0
    print("SUMMARY model={} images={} images_with_face={} total_faces={} avg_best={:.6f} min_best={:.6f} max_best={:.6f}".format(
        args.model, len(image_paths), face_images, total_faces, avg_best, min_best, max_best))

    # 释放
    rknn.release()
