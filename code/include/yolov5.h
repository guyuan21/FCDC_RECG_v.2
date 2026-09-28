// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Lightweight YOLOv5 person detector wrapper for RV1106/RV1103.

#ifndef _RKNN_DEMO_YOLOV5_H_
#define _RKNN_DEMO_YOLOV5_H_

#include <stdbool.h>

#include "common.h"
#include "rknn_api.h"
#include "postprocess.h"

typedef struct yolov5_app_context {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr *input_attrs;
    rknn_tensor_attr *output_attrs;
    rknn_tensor_mem *input_mems[1];
    rknn_tensor_mem *output_mems[YOLOV5_MAX_OUTPUTS];
    int model_channel;
    int model_width;
    int model_height;
    bool is_quant;
} yolov5_app_context_t;

int init_yolov5_model(const char *model_path, yolov5_app_context_t *app_ctx);
int release_yolov5_model(yolov5_app_context_t *app_ctx);

// Convert an RGB888 image into the zero-copy input buffer. The caller may pass
// a letterbox_t pointer when original-frame coordinate restoration is needed.
int prepare_yolov5_input(yolov5_app_context_t *app_ctx,
                         image_buffer_t *src_image,
                         letterbox_t *letterbox);

int inference_yolov5_model(yolov5_app_context_t *app_ctx,
                           yolov5_person_result_list_t *od_results);

#endif // _RKNN_DEMO_YOLOV5_H_
