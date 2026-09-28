import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


SCRFD_DIR = Path(__file__).resolve().parents[1] / 'scrfd'
sys.path.insert(0, str(SCRFD_DIR))

from scrfd import decode_outputs, parse_input_size, preprocess  # noqa: E402


ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arcface-model', default='./w600k_mbf.onnx')
    parser.add_argument('--facenet-model', default='../facenet/mobilefacenet.onnx')
    parser.add_argument('--det-model', default='../scrfd/det_25g.onnx')
    parser.add_argument('--image-dir', default='.')
    parser.add_argument('--images', nargs='*',
                        default=['001.jpg', '002.jpg', 'face15.jpg', 'face16.jpg'])
    parser.add_argument('--pair', action='append',
                        help='Pair to compare, for example 001.jpg,002.jpg. Can be repeated.')
    parser.add_argument('--det-input-size', default='640,640')
    parser.add_argument('--score-thresh', type=float, default=0.3)
    parser.add_argument('--nms-thresh', type=float, default=0.4)
    parser.add_argument('--save-aligned', action='store_true')
    parser.add_argument('--out-dir', default='./aligned112')
    return parser.parse_args()


def l2_normalize(feat):
    feat = np.asarray(feat, dtype=np.float32)
    norm = np.linalg.norm(feat, axis=1, keepdims=True)
    return feat / np.maximum(norm, 1e-12)


def compare(feat1, feat2):
    cosine = float(np.sum(feat1 * feat2, axis=1)[0])
    l2 = float(np.linalg.norm(feat1 - feat2, axis=1)[0])
    return cosine, l2


def build_pairs(images, pair_args):
    if pair_args:
        pairs = []
        for pair in pair_args:
            left, right = pair.split(',', 1)
            pairs.append((left.strip(), right.strip()))
        return pairs

    pairs = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            pairs.append((images[i], images[j]))
    return pairs


class SCRFDDetector:
    def __init__(self, model_path, input_size, score_thresh, nms_thresh):
        self.session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh

    def detect(self, img):
        infer_img, det_scale = preprocess(img, self.input_size, 'onnx')
        outputs = self.session.run(None, {self.input_name: infer_img})
        return decode_outputs(outputs, self.input_size, det_scale,
                              self.score_thresh, self.nms_thresh, img.shape)


class ArcFaceONNX:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.input_size = (112, 112)

    def get_feat_from_bgr(self, img):
        blob = cv2.dnn.blobFromImage(
            img,
            1.0 / 127.5,
            self.input_size,
            (127.5, 127.5, 127.5),
            swapRB=True,
        )
        feat = self.session.run(self.output_names, {self.input_name: blob})[0]
        return l2_normalize(feat.reshape((feat.shape[0], -1)))


class FaceNetONNX:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]
        input_shape = self.session.get_inputs()[0].shape
        self.input_size = tuple(input_shape[2:4][::-1])

    def get_feat_from_bgr(self, img):
        blob = cv2.dnn.blobFromImage(
            img,
            1.0 / 255.0,
            self.input_size,
            (0.0, 0.0, 0.0),
            swapRB=True,
        )
        feat = self.session.run(self.output_names, {self.input_name: blob})[0]
        return l2_normalize(feat.reshape((feat.shape[0], -1)))


def norm_crop_bgr(img, landmark, image_size=112):
    src = np.asarray(landmark, dtype=np.float32).reshape(5, 2)
    dst = ARCFACE_TEMPLATE.copy()
    if image_size != 112:
        dst *= float(image_size) / 112.0

    matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if matrix is None:
        raise RuntimeError('estimateAffinePartial2D failed')
    return cv2.warpAffine(img, matrix, (image_size, image_size), borderValue=0.0)


def load_and_align_images(args, detector):
    image_dir = Path(args.image_dir)
    aligned = {}
    raw = {}
    det_info = {}

    out_dir = Path(args.out_dir)
    if args.save_aligned:
        out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.images:
        path = image_dir / name
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError('failed to read image: {}'.format(path))

        dets = detector.detect(img)
        if len(dets) == 0:
            raise RuntimeError('SCRFD found no face in {}'.format(path))

        det = dets[0]
        landmark = det[5:15].reshape(5, 2)
        aligned_img = norm_crop_bgr(img, landmark, 112)
        raw[name] = img
        aligned[name] = aligned_img
        det_info[name] = det

        points = [(int(landmark[i, 0]), int(landmark[i, 1])) for i in range(5)]
        print('detect {} score={:.6f} box=({} {} {} {}) landmarks={}'.format(
            name, float(det[4]), int(det[0]), int(det[1]), int(det[2]), int(det[3]), points))

        if args.save_aligned:
            out_path = out_dir / (Path(name).stem + '_aligned112.jpg')
            cv2.imwrite(str(out_path), aligned_img)
            print('save aligned:', out_path)

    return raw, aligned, det_info


def main():
    args = parse_args()
    det_input_size = parse_input_size(args.det_input_size)
    pairs = build_pairs(args.images, args.pair)

    detector = SCRFDDetector(args.det_model, det_input_size,
                             args.score_thresh, args.nms_thresh)
    arcface = ArcFaceONNX(args.arcface_model)
    facenet = FaceNetONNX(args.facenet_model)

    print('detector:', args.det_model, 'input_size:', det_input_size,
          'score_thresh:', args.score_thresh)
    print('arcface:', args.arcface_model, 'input=112x112 aligned template')
    print('facenet:', args.facenet_model, 'input={}x{}'.format(
        facenet.input_size[0], facenet.input_size[1]))

    raw, aligned, _ = load_and_align_images(args, detector)

    arc_raw_feats = {}
    arc_aligned_feats = {}
    facenet_raw_feats = {}
    facenet_aligned_feats = {}
    for name in args.images:
        arc_raw_feats[name] = arcface.get_feat_from_bgr(raw[name])
        arc_aligned_feats[name] = arcface.get_feat_from_bgr(aligned[name])
        facenet_raw_feats[name] = facenet.get_feat_from_bgr(raw[name])
        facenet_aligned_feats[name] = facenet.get_feat_from_bgr(aligned[name])

    print('\nPair comparison:')
    print('{:<22} {:>14} {:>14} {:>14} {:>14} {:>14} {:>14} {:>14} {:>14}'.format(
        'pair',
        'arc_raw_cos', 'arc_raw_l2',
        'arc_align_cos', 'arc_align_l2',
        'fn_raw_cos', 'fn_raw_l2',
        'fn_align_cos', 'fn_align_l2'))
    for left, right in pairs:
        arc_raw_cos, arc_raw_l2 = compare(arc_raw_feats[left], arc_raw_feats[right])
        arc_align_cos, arc_align_l2 = compare(arc_aligned_feats[left], arc_aligned_feats[right])
        fn_raw_cos, fn_raw_l2 = compare(facenet_raw_feats[left], facenet_raw_feats[right])
        fn_align_cos, fn_align_l2 = compare(facenet_aligned_feats[left], facenet_aligned_feats[right])
        print('{:<22} {:>14.6f} {:>14.6f} {:>14.6f} {:>14.6f} {:>14.6f} {:>14.6f} {:>14.6f} {:>14.6f}'.format(
            '{}<->{}'.format(left, right),
            arc_raw_cos, arc_raw_l2,
            arc_align_cos, arc_align_l2,
            fn_raw_cos, fn_raw_l2,
            fn_align_cos, fn_align_l2))

    print('\nHigher cosine and lower L2 mean more similar within the same model/column.')


if __name__ == '__main__':
    main()
