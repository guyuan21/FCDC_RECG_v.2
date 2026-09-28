import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='./w600k_mbf.onnx')
    parser.add_argument('--image1', default='./001.jpg')
    parser.add_argument('--image2', default='./002.jpg')
    parser.add_argument('--image-dir', default=None)
    parser.add_argument('--input-size', type=int, default=112)
    return parser.parse_args()


def l2_normalize(feat):
    feat = np.asarray(feat, dtype=np.float32)
    norm = np.linalg.norm(feat, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    return feat / norm


def preprocess_bgr(path, input_size):
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError('failed to read image: {}'.format(path))

    # Same preprocessing as InsightFace ArcFace ONNX:
    # RGB order, NCHW layout, (pixel - 127.5) / 127.5.
    blob = cv2.dnn.blobFromImage(
        img,
        scalefactor=1.0 / 127.5,
        size=(input_size, input_size),
        mean=(127.5, 127.5, 127.5),
        swapRB=True,
    )
    return blob.astype(np.float32)


def get_embedding(session, input_name, path, input_size):
    blob = preprocess_bgr(path, input_size)
    feat = session.run(None, {input_name: blob})[0]
    feat = feat.reshape((feat.shape[0], -1))
    return l2_normalize(feat)


def compare_embeddings(feat1, feat2):
    cosine = float(np.sum(feat1 * feat2, axis=1)[0])
    distance = float(np.linalg.norm(feat1 - feat2, axis=1)[0])
    return cosine, distance


def collect_images(image_dir):
    image_dir = Path(image_dir)
    images = []
    for pattern in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        images.extend(image_dir.glob(pattern))
    return sorted(images)


def main():
    args = parse_args()

    session = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    input_name = input_meta.name

    print('model:', args.model)
    print('input:', input_name, input_meta.shape)
    print('output:', output_meta.name, output_meta.shape)
    print('preprocess: RGB NCHW, (pixel - 127.5) / 127.5, resize {}x{}'.format(
        args.input_size, args.input_size))

    if args.image_dir:
        images = collect_images(args.image_dir)
        if len(images) < 2:
            raise RuntimeError('need at least two images in {}'.format(args.image_dir))
    else:
        images = [Path(args.image1), Path(args.image2)]

    features = []
    for image in images:
        feat = get_embedding(session, input_name, image, args.input_size)
        features.append(feat)
        print('feature:', image.name, 'shape={}'.format(tuple(feat.shape)),
              'norm={:.6f}'.format(float(np.linalg.norm(feat, axis=1)[0])))

    if len(features) == 2:
        cosine, distance = compare_embeddings(features[0], features[1])
        print('compare: {} <-> {}'.format(images[0].name, images[1].name))
        print('cosine_similarity: {:.6f}'.format(cosine))
        print('l2_distance: {:.6f}'.format(distance))
        return

    print('pairwise:')
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            cosine, distance = compare_embeddings(features[i], features[j])
            print('{} <-> {} cosine={:.6f} l2={:.6f}'.format(
                images[i].name, images[j].name, cosine, distance))


if __name__ == '__main__':
    main()
