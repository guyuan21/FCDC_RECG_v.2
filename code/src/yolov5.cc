// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "yolov5.h"
#include "image_utils.h"

#define YOLOV5_VERBOSE_LOG 0

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
#if YOLOV5_VERBOSE_LOG
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
#else
    (void)attr;
#endif
}

int init_yolov5_model(const char *model_path, yolov5_app_context_t *app_ctx)
{
    int ret;
    rknn_context ctx = 0;

    if (!model_path || !app_ctx) {
        return -1;
    }
    memset(app_ctx, 0, sizeof(*app_ctx));

    ret = rknn_init(&ctx, (char *)model_path, 0, 0, NULL);
    if (ret < 0)
    {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }
    app_ctx->rknn_ctx = ctx;

    // Get Model Input Output Number
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        release_yolov5_model(app_ctx);
        return -1;
    }
    if (io_num.n_input != 1 || io_num.n_output < 1 || io_num.n_output > YOLOV5_MAX_OUTPUTS)
    {
        printf("unsupported yolov5 io count: input=%u output=%u\n",
               io_num.n_input, io_num.n_output);
        release_yolov5_model(app_ctx);
        return -1;
    }
    app_ctx->io_num = io_num;
    //printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // Get Model Input Info
    //printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            release_yolov5_model(app_ctx);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // Get Model Output Info
    //printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        //When using the zero-copy API interface, query the native output tensor attribute
        ret = rknn_query(ctx, RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            release_yolov5_model(app_ctx);
            return -1;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    // default input type is int8 (normalize and quantize need compute in outside)
    // if set uint8, will fuse normalize and quantize to npu
    input_attrs[0].type = RKNN_TENSOR_UINT8;
    // default fmt is NHWC,1106 npu only support NHWC in zero copy mode
    input_attrs[0].fmt = RKNN_TENSOR_NHWC;
    //printf("input_attrs[0].size_with_stride=%d\n", input_attrs[0].size_with_stride);
    app_ctx->input_mems[0] = rknn_create_mem(ctx, input_attrs[0].size_with_stride);
    if (!app_ctx->input_mems[0]) {
        printf("input_mems rknn_create_mem fail!\n");
        release_yolov5_model(app_ctx);
        return -1;
    }

    // Set input tensor memory
    ret = rknn_set_io_mem(ctx, app_ctx->input_mems[0], &input_attrs[0]);
    if (ret < 0) {
        printf("input_mems rknn_set_io_mem fail! ret=%d\n", ret);
        release_yolov5_model(app_ctx);
        return -1;
    }

    // Set output tensor memory
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        if (i >= YOLOV5_MAX_OUTPUTS) {
            printf("too many yolov5 output tensors: %u\n", io_num.n_output);
            release_yolov5_model(app_ctx);
            return -1;
        }
        app_ctx->output_mems[i] = rknn_create_mem(ctx, output_attrs[i].size_with_stride);
        if (!app_ctx->output_mems[i]) {
            printf("output_mems[%u] rknn_create_mem fail!\n", i);
            release_yolov5_model(app_ctx);
            return -1;
        }
        ret = rknn_set_io_mem(ctx, app_ctx->output_mems[i], &output_attrs[i]);
        if (ret < 0) {
            printf("output_mems rknn_set_io_mem fail! ret=%d\n", ret);
            release_yolov5_model(app_ctx);
            return -1;
        }
    }

    if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC)
    {
        app_ctx->is_quant = true;
    }
    else
    {
        app_ctx->is_quant = false;
    }

    app_ctx->input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    if (!app_ctx->input_attrs) {
        release_yolov5_model(app_ctx);
        return -1;
    }
    memcpy(app_ctx->input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    app_ctx->output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    if (!app_ctx->output_attrs) {
        release_yolov5_model(app_ctx);
        return -1;
    }
    memcpy(app_ctx->output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) 
    {
        app_ctx->model_channel = input_attrs[0].dims[1];
        app_ctx->model_height  = input_attrs[0].dims[2];
        app_ctx->model_width   = input_attrs[0].dims[3];
    } else 
    {
        app_ctx->model_height  = input_attrs[0].dims[1];
        app_ctx->model_width   = input_attrs[0].dims[2];
        app_ctx->model_channel = input_attrs[0].dims[3];
    } 

#if YOLOV5_VERBOSE_LOG
    printf("model input height=%d, width=%d, channel=%d\n",
           app_ctx->model_height, app_ctx->model_width, app_ctx->model_channel);
#endif

    return 0;
}

int release_yolov5_model(yolov5_app_context_t *app_ctx)
{
    if (!app_ctx) return 0;

    for (int i = 0; i < app_ctx->io_num.n_input && i < 1; i++) {
        if (app_ctx->input_mems[i] != NULL) {
            rknn_destroy_mem(app_ctx->rknn_ctx, app_ctx->input_mems[i]);
            app_ctx->input_mems[i] = NULL;
        }
    }
    for (int i = 0; i < app_ctx->io_num.n_output && i < YOLOV5_MAX_OUTPUTS; i++) {
        if (app_ctx->output_mems[i] != NULL) {
            rknn_destroy_mem(app_ctx->rknn_ctx, app_ctx->output_mems[i]);
            app_ctx->output_mems[i] = NULL;
        }
    }
    if (app_ctx->input_attrs != NULL)
    {
        free(app_ctx->input_attrs);
        app_ctx->input_attrs = NULL;
    }
    if (app_ctx->output_attrs != NULL)
    {
        free(app_ctx->output_attrs);
        app_ctx->output_attrs = NULL;
    }
    if (app_ctx->rknn_ctx != 0)
    {
        rknn_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    return 0;
}

int prepare_yolov5_input(yolov5_app_context_t *app_ctx,
                         image_buffer_t *src_image,
                         letterbox_t *letterbox)
{
    if (!app_ctx || !src_image || !app_ctx->input_mems[0]) {
        return -1;
    }
    if (src_image->format != IMAGE_FORMAT_RGB888) {
        printf("prepare_yolov5_input only supports RGB888, fmt=%d\n", src_image->format);
        return -1;
    }

    image_buffer_t dst_image;
    memset(&dst_image, 0, sizeof(dst_image));
    dst_image.width = app_ctx->model_width;
    dst_image.height = app_ctx->model_height;
    dst_image.width_stride = app_ctx->model_width;
    dst_image.height_stride = app_ctx->model_height;
    dst_image.format = IMAGE_FORMAT_RGB888;
    dst_image.virt_addr = (unsigned char *)app_ctx->input_mems[0]->virt_addr;
    dst_image.size = app_ctx->input_attrs[0].size_with_stride;
    dst_image.fd = app_ctx->input_mems[0]->fd;

    return convert_image_with_letterbox(src_image, &dst_image, letterbox, 114);
}

int inference_yolov5_model(yolov5_app_context_t *app_ctx,
                           yolov5_person_result_list_t *od_results)
{
    int ret;
    const float nms_threshold = YOLOV5_NMS_THRESH;
    const float box_conf_threshold = YOLOV5_BOX_THRESH;
   
    if (!app_ctx || !od_results) {
        return -1;
    }

    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }

    // Post Process
    post_process(app_ctx, app_ctx->output_mems,  box_conf_threshold, nms_threshold, od_results);
    return ret;
}
