import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arcface-model', default='./w600k_mbf.onnx')
    parser.add_argument('--facenet-model', default='../facenet/mobilefacenet.onnx')
    parser.add_argument('--image-dir', default='.')
    parser.add_argument('--images', nargs='*',
                        default=['001.jpg', '002.jpg', 'face15.jpg', 'face16.jpg'])
    parser.add_argument('--pair', action='append',
                        help='Pair to compare, for example 001.jpg,002.jpg. Can be repeated.')
    return parser.parse_args()


def l2_normalize(feat):
    feat = np.asarray(feat, dtype=np.float32)
    norm = np.linalg.norm(feat, axis=1, keepdims=True)
    return feat / np.maximum(norm, 1e-12)


class ArcFaceONNX:
    """Minimal InsightFace arcface_onnx.py compatible inference wrapper."""

    def __init__(self, model_path):
        self.session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.input_size = tuple(input_shape[2:4][::-1])
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.input_mean = 127.5
        self.input_std = 127.5

    def get_feat(self, image_path):
        img = cv2.imread(str(image_path))
        if img is None:
            raise RuntimeError('failed to read image: {}'.format(image_path))
        blob = cv2.dnn.blobFromImage(
            img,
            1.0 / self.input_std,
            self.input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True,
        )
        feat = self.session.run(self.output_names, {self.input_name: blob})[0]
        return l2_normalize(feat.reshape((feat.shape[0], -1)))


class FaceNetONNX:
    """Old project MobileFaceNet/FaceNet wrapper matching facenet.py preprocessing."""

    def __init__(self, model_path):
        self.session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.input_size = tuple(input_shape[2:4][::-1])
        self.output_names = [out.name for out in self.session.get_outputs()]

    def get_feat(self, image_path):
        img = cv2.imread(str(image_path))
        if img is None:
            raise RuntimeError('failed to read image: {}'.format(image_path))
        # facenet.py feeds RGB NHWC uint8 to RKNN with mean=0/std=255.
        # For ONNXRuntime we provide equivalent RGB NCHW float32 in [0, 1].
        blob = cv2.dnn.blobFromImage(
            img,
            1.0 / 255.0,
            self.input_size,
            (0.0, 0.0, 0.0),
            swapRB=True,
        )
        feat = self.session.run(self.output_names, {self.input_name: blob})[0]
        return l2_normalize(feat.reshape((feat.shape[0], -1)))


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


def main():
    args = parse_args()
    image_dir = Path(args.image_dir)
    images = args.images
    pairs = build_pairs(images, args.pair)

    arcface = ArcFaceONNX(args.arcface_model)
    facenet = FaceNetONNX(args.facenet_model)

    print('ArcFace/MobileFace model:', args.arcface_model)
    print('  input:', arcface.session.get_inputs()[0].shape,
          'output:', arcface.session.get_outputs()[0].shape)
    print('  preprocess: InsightFace arcface_onnx.py, RGB NCHW, (pixel-127.5)/127.5')
    print('FaceNet model:', args.facenet_model)
    print('  input:', facenet.session.get_inputs()[0].shape,
          'output:', facenet.session.get_outputs()[0].shape)
    print('  preprocess: RGB NCHW, pixel/255')

    arc_feats = {}
    face_feats = {}
    for name in images:
        path = image_dir / name
        arc_feats[name] = arcface.get_feat(path)
        face_feats[name] = facenet.get_feat(path)
        print('feature {} arcface_norm={:.6f} facenet_norm={:.6f}'.format(
            name,
            float(np.linalg.norm(arc_feats[name], axis=1)[0]),
            float(np.linalg.norm(face_feats[name], axis=1)[0]),
        ))

    print('\nPair comparison:')
    print('{:<22} {:>16} {:>14} {:>16} {:>14}'.format(
        'pair', 'arc_cosine', 'arc_l2', 'facenet_cosine', 'facenet_l2'))
    for left, right in pairs:
        arc_cos, arc_l2 = compare(arc_feats[left], arc_feats[right])
        face_cos, face_l2 = compare(face_feats[left], face_feats[right])
        print('{:<22} {:>16.6f} {:>14.6f} {:>16.6f} {:>14.6f}'.format(
            '{}<->{}'.format(left, right),
            arc_cos, arc_l2,
            face_cos, face_l2,
        ))

    print('\nNote: compare same-model score distributions. ArcFace and FaceNet thresholds are not interchangeable.')


if __name__ == '__main__':
    main()
