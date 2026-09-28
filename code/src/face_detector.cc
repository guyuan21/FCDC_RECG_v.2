// =============================================================================
// face_detector.cc  —  SCRFD face detector and shared face model runtime init
// Platform : Luckfox Pico Pro Max (RV1106)
// =============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "face_models.h"

// USE_RKNN_MEM_SHARE=1：两个模型共享最大那块内部内存，节省SRAM
// Pro Max 有更多内存，可按需开启
#define USE_RKNN_MEM_SHARE 1
#define FACE_MODEL_VERBOSE_LOG 0

// =============================================================================
// 工具函数
// =============================================================================
static inline int clamp_i(int x, int lo, int hi)
{
    return x < lo ? lo : (x > hi ? hi : x);
}

static void cleanup_rknn_context(rknn_app_context_t *app_ctx)
{
    if (!app_ctx) return;
    for (int i = 0; i < RKNN_MAX_IO; ++i) {
        if (app_ctx->input_mems[i] && app_ctx->rknn_ctx) {
            rknn_destroy_mem(app_ctx->rknn_ctx, app_ctx->input_mems[i]);
            app_ctx->input_mems[i] = nullptr;
        }
        if (app_ctx->output_mems[i] && app_ctx->rknn_ctx) {
            rknn_destroy_mem(app_ctx->rknn_ctx, app_ctx->output_mems[i]);
            app_ctx->output_mems[i] = nullptr;
        }
    }
    if (app_ctx->net_mem && app_ctx->rknn_ctx) {
        rknn_destroy_mem(app_ctx->rknn_ctx, app_ctx->net_mem);
        app_ctx->net_mem = nullptr;
    }
    if (app_ctx->max_mem && app_ctx->rknn_ctx) {
        rknn_destroy_mem(app_ctx->rknn_ctx, app_ctx->max_mem);
        app_ctx->max_mem = nullptr;
    }
    free(app_ctx->input_attrs);
    free(app_ctx->output_attrs);
    app_ctx->input_attrs = nullptr;
    app_ctx->output_attrs = nullptr;
    if (app_ctx->rknn_ctx) {
        rknn_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
}

static void dump_tensor_attr(const char *tag, rknn_tensor_attr *attr)
{
#if FACE_MODEL_VERBOSE_LOG
    printf("  [%s] index=%d name=%s dims=[%d,%d,%d,%d] fmt=%s type=%s qnt=%s zp=%d scale=%f\n",
           tag, attr->index, attr->name,
           attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           get_format_string(attr->fmt),
           get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type),
           attr->zp, attr->scale);
#else
    (void)tag;
    (void)attr;
#endif
}

static void parse_input_dims(const rknn_tensor_attr &attr,
                             int *height,
                             int *width,
                             int *channel)
{
    if (attr.n_dims == 4 && attr.dims[1] <= 4 &&
        attr.dims[2] > 16 && attr.dims[3] > 16) {
        if (height) *height = attr.dims[2];
        if (width) *width = attr.dims[3];
        if (channel) *channel = attr.dims[1];
        return;
    }

    if (height) *height = attr.dims[1];
    if (width) *width = attr.dims[2];
    if (channel) *channel = attr.dims[3];
}

// 通用：为一个 rknn_context 查询并设置 zero-copy IO 内存
static int setup_rknn_io_mem(rknn_context ctx,
                              rknn_app_context_t *app_ctx,
                              const char *model_tag)
{
    rknn_input_output_num io_num;
    int ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) { printf("[%s] query io_num fail\n", model_tag); return -1; }
    if (io_num.n_input > RKNN_MAX_IO || io_num.n_output > RKNN_MAX_IO) {
        printf("[%s] io count overflow input=%u output=%u\n",
               model_tag, io_num.n_input, io_num.n_output);
        return -1;
    }
#if FACE_MODEL_VERBOSE_LOG
    printf("[%s] inputs=%d outputs=%d\n", model_tag, io_num.n_input, io_num.n_output);
#endif

    // 输入属性
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (uint32_t i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &input_attrs[i], sizeof(rknn_tensor_attr));
        dump_tensor_attr(model_tag, &input_attrs[i]);
    }

    // 输出属性
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (uint32_t i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr));
        dump_tensor_attr(model_tag, &output_attrs[i]);
    }

    // 输入 zero-copy 内存（UINT8 NHWC）
    input_attrs[0].type = RKNN_TENSOR_UINT8;
    input_attrs[0].fmt  = RKNN_TENSOR_NHWC;
    app_ctx->input_mems[0] = rknn_create_mem(ctx, input_attrs[0].size_with_stride);
    if (!app_ctx->input_mems[0]) { printf("[%s] create input mem fail\n", model_tag); return -1; }
    ret = rknn_set_io_mem(ctx, app_ctx->input_mems[0], &input_attrs[0]);
    if (ret < 0) { printf("[%s] set input mem fail\n", model_tag); return -1; }

    // 输出 zero-copy 内存
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        app_ctx->output_mems[i] = rknn_create_mem(ctx, output_attrs[i].size_with_stride);
        if (!app_ctx->output_mems[i]) {
            printf("[%s] create output mem[%d] fail\n", model_tag, i);
            return -1;
        }
        ret = rknn_set_io_mem(ctx, app_ctx->output_mems[i], &output_attrs[i]);
        if (ret < 0) { printf("[%s] set output mem[%d] fail\n", model_tag, i); return -1; }
    }

    app_ctx->rknn_ctx  = ctx;
    app_ctx->io_num    = io_num;
    app_ctx->is_quant  = (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC);

    app_ctx->input_attrs  = (rknn_tensor_attr *)malloc(io_num.n_input  * sizeof(rknn_tensor_attr));
    app_ctx->output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    if (!app_ctx->input_attrs || !app_ctx->output_attrs) {
        printf("[%s] alloc tensor attrs fail\n", model_tag);
        return -1;
    }
    memcpy(app_ctx->input_attrs,  input_attrs,  io_num.n_input  * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    parse_input_dims(input_attrs[0],
                     &app_ctx->model_height,
                     &app_ctx->model_width,
                     &app_ctx->model_channel);

    return 0;
}

// =============================================================================
// 双模型联合初始化（支持 MEM_SHARE）
// =============================================================================
int init_face_models(const char *retina_path,
                                  const char *facenet_path,
                                  rknn_app_context_t *retina_ctx,
                                  rknn_app_context_t *facenet_ctx)
{
    int ret;
    rknn_context ctx_retina  = 0;
    rknn_context ctx_facenet = 0;

#if USE_RKNN_MEM_SHARE
    // ----------------------------------------------------------------
    // MEM_SHARE：两模型共用一块最大内部内存，交替推理时节省SRAM
    // ----------------------------------------------------------------
    ret = rknn_init(&ctx_retina,  (char *)retina_path,  0, RKNN_FLAG_MEM_ALLOC_OUTSIDE, NULL);
    if (ret < 0) { printf("[Init] retina rknn_init fail\n"); return -1; }
    retina_ctx->rknn_ctx = ctx_retina;
    ret = rknn_init(&ctx_facenet, (char *)facenet_path, 0, RKNN_FLAG_MEM_ALLOC_OUTSIDE, NULL);
    if (ret < 0) {
        printf("[Init] facenet rknn_init fail\n");
        rknn_destroy(ctx_retina);
        retina_ctx->rknn_ctx = 0;
        return -1;
    }
    facenet_ctx->rknn_ctx = ctx_facenet;

    rknn_mem_size ms_retina, ms_facenet;
    rknn_query(ctx_retina,  RKNN_QUERY_MEM_SIZE, &ms_retina,  sizeof(ms_retina));
    rknn_query(ctx_facenet, RKNN_QUERY_MEM_SIZE, &ms_facenet, sizeof(ms_facenet));

    uint32_t shared_size = (ms_retina.total_internal_size > ms_facenet.total_internal_size)
                           ? ms_retina.total_internal_size : ms_facenet.total_internal_size;
#if FACE_MODEL_VERBOSE_LOG
    printf("[Init] MEM_SHARE size=%u bytes\n", shared_size);
#endif

    // 分配最大内存块（挂在 retina 上）
    retina_ctx->max_mem = rknn_create_mem(ctx_retina, shared_size);
    if (!retina_ctx->max_mem) {
        printf("[Init] shared mem alloc fail\n");
        rknn_destroy(ctx_facenet);
        rknn_destroy(ctx_retina);
        return -1;
    }

    retina_ctx->net_mem = rknn_create_mem_from_fd(
        ctx_retina,
        retina_ctx->max_mem->fd,
        retina_ctx->max_mem->virt_addr,
        ms_retina.total_internal_size, 0);
    if (!retina_ctx->net_mem) {
        printf("[Init] retina net_mem create fail\n");
        cleanup_rknn_context(retina_ctx);
        rknn_destroy(ctx_facenet);
        return -1;
    }
    rknn_set_internal_mem(ctx_retina, retina_ctx->net_mem);

    facenet_ctx->net_mem = rknn_create_mem_from_fd(
        ctx_facenet,
        retina_ctx->max_mem->fd,
        retina_ctx->max_mem->virt_addr,   // 从同一块起始复用
        ms_facenet.total_internal_size, 0);
    if (!facenet_ctx->net_mem) {
        printf("[Init] facenet net_mem create fail\n");
        cleanup_rknn_context(retina_ctx);
        rknn_destroy(ctx_facenet);
        return -1;
    }
    rknn_set_internal_mem(ctx_facenet, facenet_ctx->net_mem);

#else
    // ----------------------------------------------------------------
    // 独立内存（调试/兼容模式）
    // ----------------------------------------------------------------
    ret = rknn_init(&ctx_retina,  (char *)retina_path,  0, 0, NULL);
    if (ret < 0) { printf("[Init] retina rknn_init fail\n"); return -1; }
    retina_ctx->rknn_ctx = ctx_retina;
    ret = rknn_init(&ctx_facenet, (char *)facenet_path, 0, 0, NULL);
    if (ret < 0) {
        printf("[Init] facenet rknn_init fail\n");
        rknn_destroy(ctx_retina);
        retina_ctx->rknn_ctx = 0;
        return -1;
    }
    facenet_ctx->rknn_ctx = ctx_facenet;
#endif

    // 配置 IO 内存
    if (setup_rknn_io_mem(ctx_retina,  retina_ctx,  "SCRFD") < 0) {
        cleanup_rknn_context(facenet_ctx);
        cleanup_rknn_context(retina_ctx);
        return -1;
    }
    if (setup_rknn_io_mem(ctx_facenet, facenet_ctx, "MobileFace") < 0) {
        cleanup_rknn_context(facenet_ctx);
        cleanup_rknn_context(retina_ctx);
        return -1;
    }

#if FACE_MODEL_VERBOSE_LOG
    printf("[Init] Both models ready.\n");
#endif
    return 0;
}

// =============================================================================
// 释放人脸检测模型（含 max_mem）
// =============================================================================
int release_face_detector(rknn_app_context_t *app_ctx)
{
    for (int i = 0; i < (int)app_ctx->io_num.n_input; i++) {
        if (app_ctx->input_mems[i]) {
            rknn_destroy_mem(app_ctx->rknn_ctx, app_ctx->input_mems[i]);
            app_ctx->input_mems[i] = nullptr;
        }
    }
    for (int i = 0; i < (int)app_ctx->io_num.n_output; i++) {
        if (app_ctx->output_mems[i]) {
            rknn_destroy_mem(app_ctx->rknn_ctx, app_ctx->output_mems[i]);
            app_ctx->output_mems[i] = nullptr;
        }
    }
    if (app_ctx->net_mem) {
        rknn_destroy_mem(app_ctx->rknn_ctx, app_ctx->net_mem);
        app_ctx->net_mem = nullptr;
    }
    if (app_ctx->max_mem) {
        rknn_destroy_mem(app_ctx->rknn_ctx, app_ctx->max_mem);
        app_ctx->max_mem = nullptr;
    }
    free(app_ctx->input_attrs);  app_ctx->input_attrs  = nullptr;
    free(app_ctx->output_attrs); app_ctx->output_attrs = nullptr;
    if (app_ctx->rknn_ctx) {
        rknn_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    return 0;
}

// =============================================================================
// NMS 相关工具
// =============================================================================
static float calc_iou(float x0, float y0, float x1, float y1,
                      float x2, float y2, float x3, float y3)
{
    float w = fmaxf(0.f, fminf(x1, x3) - fmaxf(x0, x2) + 1.f);
    float h = fmaxf(0.f, fminf(y1, y3) - fmaxf(y0, y2) + 1.f);
    float inter = w * h;
    float uni = (x1-x0+1)*(y1-y0+1) + (x3-x2+1)*(y3-y2+1) - inter;
    return (uni <= 0.f) ? 0.f : inter / uni;
}

// 快速降序排列（按 scores 值）
static int qsort_desc(float *scores, int left, int right, int *indices)
{
    int low = left, high = right;
    float pivot = scores[left];
    int   pidx  = indices[left];
    while (low < high) {
        while (low < high && scores[high] <= pivot) high--;
        scores[low] = scores[high]; indices[low] = indices[high];
        while (low < high && scores[low]  >= pivot) low++;
        scores[high] = scores[low]; indices[high] = indices[low];
    }
    scores[low] = pivot; indices[low] = pidx;
    if (left  < low-1) qsort_desc(scores, left,   low-1, indices);
    if (low+1 < right) qsort_desc(scores, low+1,  right, indices);
    return low;
}

static void nms(int count, float *locs, int *order, float thresh)
{
    for (int i = 0; i < count; i++) {
        if (order[i] == -1) continue;
        int n = order[i];
        for (int j = i+1; j < count; j++) {
            int m = order[j];
            if (m == -1) continue;
            float iou = calc_iou(locs[n*4+0], locs[n*4+1], locs[n*4+2], locs[n*4+3],
                                 locs[m*4+0], locs[m*4+1], locs[m*4+2], locs[m*4+3]);
            if (iou > thresh) order[j] = -1;
        }
    }
}

static inline float deqnt_i32_f32(int32_t q, int32_t zp, float sc)
{
    return ((float)q - (float)zp) * sc;
}

static float output_elem_f32(rknn_app_context_t *app_ctx, int out_idx, int elem_idx)
{
    rknn_tensor_attr *attr = &app_ctx->output_attrs[out_idx];
    void *base = app_ctx->output_mems[out_idx]->virt_addr;

    if (attr->type == RKNN_TENSOR_FLOAT32) {
        return ((float *)base)[elem_idx];
    }
    if (attr->type == RKNN_TENSOR_INT8) {
        return deqnt_i32_f32((int32_t)((int8_t *)base)[elem_idx], attr->zp, attr->scale);
    }
    if (attr->type == RKNN_TENSOR_UINT8) {
        return deqnt_i32_f32((int32_t)((uint8_t *)base)[elem_idx], attr->zp, attr->scale);
    }

    return 0.f;
}

static int decode_scrfd_outputs(rknn_app_context_t *app_ctx,
                                object_detect_result_list *od_results)
{
    const int STRIDES[3] = {8, 16, 32};
    const int NUM_ANCHORS = 2;
    const int MAX_PRIORS = 16800;
    const float CONF_THRESH = MIN_FACE_DETECT_SCORE;
    const float NMS_THRESH = 0.40f;

    static float loc_fp32[MAX_PRIORS * 4];
    static float lnd_fp32[MAX_PRIORS * 10];
    static float props[MAX_PRIORS];
    static int   fidx[MAX_PRIORS];

    int valid = 0;
    int mw = app_ctx->model_width;
    int mh = app_ctx->model_height;

    for (int level = 0; level < 3; ++level) {
        const int stride = STRIDES[level];
        const int feat_w = mw / stride;
        const int feat_h = mh / stride;
        const int expected = feat_w * feat_h * NUM_ANCHORS;
        const int score_idx = level;
        const int bbox_idx = level + 3;
        const int kps_idx = level + 6;

        int count = expected;
        int score_count = (int)app_ctx->output_attrs[score_idx].n_elems;
        int bbox_count = (int)app_ctx->output_attrs[bbox_idx].n_elems / 4;
        int kps_count = (int)app_ctx->output_attrs[kps_idx].n_elems / 10;
        if (score_count < count) count = score_count;
        if (bbox_count < count) count = bbox_count;
        if (kps_count < count) count = kps_count;

        for (int i = 0; i < count && valid < MAX_PRIORS; ++i) {
            float sc = output_elem_f32(app_ctx, score_idx, i);
            if (sc < CONF_THRESH) continue;

            int anchor_idx = i / NUM_ANCHORS;
            float cx = (float)(anchor_idx % feat_w) * (float)stride;
            float cy = (float)(anchor_idx / feat_w) * (float)stride;

            float left   = output_elem_f32(app_ctx, bbox_idx, i * 4 + 0) * stride;
            float top    = output_elem_f32(app_ctx, bbox_idx, i * 4 + 1) * stride;
            float right  = output_elem_f32(app_ctx, bbox_idx, i * 4 + 2) * stride;
            float bottom = output_elem_f32(app_ctx, bbox_idx, i * 4 + 3) * stride;

            loc_fp32[valid * 4 + 0] = fmaxf(0.f, cx - left);
            loc_fp32[valid * 4 + 1] = fmaxf(0.f, cy - top);
            loc_fp32[valid * 4 + 2] = fminf((float)mw, cx + right);
            loc_fp32[valid * 4 + 3] = fminf((float)mh, cy + bottom);

            for (int j = 0; j < 5; ++j) {
                float px = cx + output_elem_f32(app_ctx, kps_idx, i * 10 + j * 2) * stride;
                float py = cy + output_elem_f32(app_ctx, kps_idx, i * 10 + j * 2 + 1) * stride;
                lnd_fp32[valid * 10 + j * 2] = fminf(fmaxf(px, 0.f), (float)mw);
                lnd_fp32[valid * 10 + j * 2 + 1] = fminf(fmaxf(py, 0.f), (float)mh);
            }

            props[valid] = sc;
            fidx[valid] = valid;
            valid++;
        }
    }

    if (valid > 1) qsort_desc(props, 0, valid - 1, fidx);
    nms(valid, loc_fp32, fidx, NMS_THRESH);

    int face_cnt = 0;
    for (int i = 0; i < valid && face_cnt < MAX_DETECT_FACES; i++) {
        if (fidx[i] == -1 || props[i] < CONF_THRESH) continue;
        int n = fidx[i];

        od_results->results[face_cnt].box.left =
            clamp_i((int)(loc_fp32[n * 4 + 0] + 0.5f), 0, mw);
        od_results->results[face_cnt].box.top =
            clamp_i((int)(loc_fp32[n * 4 + 1] + 0.5f), 0, mh);
        od_results->results[face_cnt].box.right =
            clamp_i((int)(loc_fp32[n * 4 + 2] + 0.5f), 0, mw);
        od_results->results[face_cnt].box.bottom =
            clamp_i((int)(loc_fp32[n * 4 + 3] + 0.5f), 0, mh);
        od_results->results[face_cnt].prop = props[i];

        for (int j = 0; j < 5; j++) {
            od_results->results[face_cnt].point[j].x =
                clamp_i((int)(lnd_fp32[n * 10 + j * 2] + 0.5f), 0, mw);
            od_results->results[face_cnt].point[j].y =
                clamp_i((int)(lnd_fp32[n * 10 + j * 2 + 1] + 0.5f), 0, mh);
        }
        face_cnt++;
    }

    od_results->count = face_cnt;
    return 0;
}

// =============================================================================
// SCRFD 推理
// =============================================================================
int run_face_detector(rknn_app_context_t *app_ctx,
                                object_detect_result_list *od_results)
{
    od_results->count = 0;
    int ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    if (ret < 0) { printf("[SCRFD] rknn_run fail ret=%d\n", ret); return -1; }

    if (app_ctx->io_num.n_output != 9) {
        printf("[SCRFD] unsupported output count=%u, expect 9\n", app_ctx->io_num.n_output);
        return -1;
    }
    return decode_scrfd_outputs(app_ctx, od_results);
}

// =============================================================================
// 坐标映射：letterbox 模型坐标 → 原图坐标
// =============================================================================
bool map_coordinates(const image_transform_t &transform, int *x, int *y)
{
    if (!x || !y || transform.scale <= 0.f ||
        transform.src_w <= 0 || transform.src_h <= 0) {
        return false;
    }

    float px = ((float)*x - transform.pad_left) / transform.scale;
    float py = ((float)*y - transform.pad_top) / transform.scale;
    *x = clamp_i((int)px, 0, transform.src_w - 1);
    *y = clamp_i((int)py, 0, transform.src_h - 1);
    return true;
}

void scale_coordinates(const cv::Size &src_space, const cv::Size &dst_space,
                       int *x, int *y)
{
    if (!x || !y || src_space.width <= 0 || src_space.height <= 0) return;
    *x = (int)((float)*x * dst_space.width / src_space.width);
    *y = (int)((float)*y * dst_space.height / src_space.height);
}
