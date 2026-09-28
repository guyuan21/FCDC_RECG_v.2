import sys

from rknn.api import RKNN

def parse_arg():
    if len(sys.argv) < 5:
        print("Usage: python3 {} [onnx_model_path] [dataset_path] [output_rknn_path] [model_type]".format(sys.argv[0]));
        exit(1)

    model_path = sys.argv[1]
    dataset_path= sys.argv[2]
    output_path = sys.argv[3]
    model_type = sys.argv[4]

    return model_path, dataset_path, output_path, model_type


def config_model(rknn, model_type):
    model_type = model_type.lower()
    inputs = None
    input_size_list = None

    if model_type == 'retinaface':
        rknn.config(mean_values=[[104, 117, 123]], std_values=[[1, 1, 1]],
                    target_platform='rv1106', quantized_algorithm='normal',
                    quant_img_RGB2BGR=True, optimization_level=0)
        print("Use retinaface mode")
    elif model_type == 'scrfd':
        rknn.config(mean_values=[[127.5, 127.5, 127.5]],
                    std_values=[[128.0, 128.0, 128.0]],
                    target_platform='rv1106', quantized_algorithm='normal',
                    optimization_level=0)
        inputs = ['input.1']
        input_size_list = [[1, 3, 640, 640]]
        print("Use SCRFD mode, fixed input 1x3x640x640")
    elif model_type in ('mobileface', 'w600k_mbf', 'arcface'):
        rknn.config(mean_values=[[127.5, 127.5, 127.5]],
                    std_values=[[127.5, 127.5, 127.5]],
                    target_platform='rv1106', quantized_algorithm='normal',
                    optimization_level=0)
        inputs = ['input.1']
        input_size_list = [[1, 3, 112, 112]]
        print("Use MobileFace/ArcFace mode, fixed input 1x3x112x112")
    else:
        rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                    target_platform='rv1106')
        print("Use facenet/default mode")

    return inputs, input_size_list

if __name__ == '__main__':
    model_path, dataset_path, output_path, model_type = parse_arg()

    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Pre-process config
    print('--> Config model')
    inputs, input_size_list = config_model(rknn, model_type)
    print('done')

    # Load model
    print('--> Loading model')
    if inputs is not None and input_size_list is not None:
        ret = rknn.load_onnx(model=model_path, inputs=inputs,
                             input_size_list=input_size_list)
    else:
        ret = rknn.load_onnx(model=model_path)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset=dataset_path)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Release
    rknn.release()
