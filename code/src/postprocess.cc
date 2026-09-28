// Person-only YOLOv5 postprocess for RV1106/RV1103.

#include "postprocess.h"
#include "yolov5.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <vector>

#define YOLOV5_POSTPROCESS_VERBOSE_LOG 0

static const int kAnchor[YOLOV5_OUTPUT_BRANCHES][6] = {
    {10, 13, 16, 30, 33, 23},
    {30, 61, 62, 45, 59, 119},
    {116, 90, 156, 198, 373, 326},
};

static inline int clamp_i(float val, int min_v, int max_v)
{
    return (int)(val > min_v ? (val < max_v ? val : max_v) : min_v);
}

static inline int32_t clip_i8(float val)
{
    if (val < -128.f) return -128;
    if (val > 127.f) return 127;
    return (int32_t)val;
}

static inline int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
    return (int8_t)clip_i8((f32 / scale) + zp);
}

static inline float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale)
{
    return ((float)qnt - (float)zp) * scale;
}

static inline float deqnt_output_to_f32(const uint8_t *data,
                                        int offset,
                                        const rknn_tensor_attr &attr)
{
    if (attr.type == RKNN_TENSOR_FLOAT32) {
        return ((const float *)data)[offset];
    }
    if (attr.type == RKNN_TENSOR_UINT8) {
        return ((float)data[offset] - (float)attr.zp) * attr.scale;
    }
    if (attr.type == RKNN_TENSOR_INT8) {
        return ((float)((const int8_t *)data)[offset] - (float)attr.zp) * attr.scale;
    }

    return 0.f;
}

static float calc_iou(float xmin0, float ymin0, float xmax0, float ymax0,
                      float xmin1, float ymin1, float xmax1, float ymax1)
{
    float w = fmaxf(0.f, fminf(xmax0, xmax1) - fmaxf(xmin0, xmin1) + 1.f);
    float h = fmaxf(0.f, fminf(ymax0, ymax1) - fmaxf(ymin0, ymin1) + 1.f);
    float inter = w * h;
    float area0 = (xmax0 - xmin0 + 1.f) * (ymax0 - ymin0 + 1.f);
    float area1 = (xmax1 - xmin1 + 1.f) * (ymax1 - ymin1 + 1.f);
    float uni = area0 + area1 - inter;
    return uni <= 0.f ? 0.f : inter / uni;
}

static void nms_person(const std::vector<float> &boxes,
                       const std::vector<float> &scores,
                       std::vector<int> &order,
                       float threshold)
{
    for (size_t i = 0; i < order.size(); ++i) {
        int n = order[i];
        if (n < 0) continue;

        float x0 = boxes[n * 4 + 0];
        float y0 = boxes[n * 4 + 1];
        float x1 = x0 + boxes[n * 4 + 2];
        float y1 = y0 + boxes[n * 4 + 3];

        for (size_t j = i + 1; j < order.size(); ++j) {
            int m = order[j];
            if (m < 0) continue;

            float xx0 = boxes[m * 4 + 0];
            float yy0 = boxes[m * 4 + 1];
            float xx1 = xx0 + boxes[m * 4 + 2];
            float yy1 = yy0 + boxes[m * 4 + 3];

            if (calc_iou(x0, y0, x1, y1, xx0, yy0, xx1, yy1) > threshold) {
                order[j] = -1;
            }
        }
    }
}

static int process_person_i8_rv1106(int8_t *input,
                                    const int *anchor,
                                    int grid_h,
                                    int grid_w,
                                    int stride,
                                    std::vector<float> &boxes,
                                    std::vector<float> &scores,
                                    float threshold,
                                    int32_t zp,
                                    float scale)
{
    int valid_count = 0;
    int8_t conf_threshold_i8 = qnt_f32_to_affine(threshold, zp, scale);
    const int anchor_per_branch = 3;
    const int align_c = YOLOV5_PROP_BOX_SIZE * anchor_per_branch;

    for (int h = 0; h < grid_h; ++h) {
        for (int w = 0; w < grid_w; ++w) {
            for (int a = 0; a < anchor_per_branch; ++a) {
                int hw_offset = h * grid_w * align_c + w * align_c + a * YOLOV5_PROP_BOX_SIZE;
                int8_t *hw_ptr = input + hw_offset;
                int8_t box_confidence = hw_ptr[4];
                if (box_confidence < conf_threshold_i8) continue;

                int8_t person_prob_i8 = hw_ptr[5 + YOLOV5_PERSON_CLASS_ID];
                float box_conf = deqnt_affine_to_f32(box_confidence, zp, scale);
                float person_prob = deqnt_affine_to_f32(person_prob_i8, zp, scale);
                float score = box_conf * person_prob;
                if (score < threshold) continue;

                float box_x = deqnt_affine_to_f32(hw_ptr[0], zp, scale) * 2.f - 0.5f;
                float box_y = deqnt_affine_to_f32(hw_ptr[1], zp, scale) * 2.f - 0.5f;
                float box_w = deqnt_affine_to_f32(hw_ptr[2], zp, scale) * 2.f;
                float box_h = deqnt_affine_to_f32(hw_ptr[3], zp, scale) * 2.f;

                box_x = (box_x + (float)w) * (float)stride;
                box_y = (box_y + (float)h) * (float)stride;
                box_w = box_w * box_w * (float)anchor[a * 2];
                box_h = box_h * box_h * (float)anchor[a * 2 + 1];

                boxes.push_back(box_x - box_w * 0.5f);
                boxes.push_back(box_y - box_h * 0.5f);
                boxes.push_back(box_w);
                boxes.push_back(box_h);
                scores.push_back(score);
                valid_count++;
            }
        }
    }

    return valid_count;
}

static int process_person_single_output_quant(uint8_t *input,
                                              const rknn_tensor_attr &attr,
                                              std::vector<float> &boxes,
                                              std::vector<float> &scores,
                                              float threshold,
                                              int model_w,
                                              int model_h)
{
    if (!input) return 0;

    const int box_size = YOLOV5_PROP_BOX_SIZE;
    const int ndims = attr.n_dims > 0 ? attr.n_dims : 0;
    int dims[16] = {0};
    int strides[16] = {0};
    int proposal_axes[16] = {0};
    int proposal_axis_count = 0;
    int class_dim = -1;
    int proposal_count = 0;
    bool flat_layout = false;

    for (int i = 0; i < ndims && i < 16; ++i) {
        dims[i] = attr.dims[i] > 0 ? attr.dims[i] : 1;
        if (dims[i] == box_size && class_dim < 0) {
            class_dim = i;
        }
    }

    if (class_dim >= 0) {
        strides[ndims - 1] = 1;
        for (int i = ndims - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * dims[i + 1];
        }

        proposal_count = 1;
        for (int i = 0; i < ndims; ++i) {
            if (i == class_dim) continue;
            if (i == 0 && dims[i] == 1) continue; // batch
            if (dims[i] == 1) continue;
            proposal_axes[proposal_axis_count++] = i;
            proposal_count *= dims[i];
        }
    } else if (attr.n_elems > 0 && attr.n_elems % box_size == 0) {
        flat_layout = true;
        proposal_count = attr.n_elems / box_size;
    }

    if (proposal_count <= 0) {
        printf("[YOLOv5] unsupported single output shape n_dims=%d dims=[%d,%d,%d,%d] elems=%d\n",
               attr.n_dims, attr.dims[0], attr.dims[1], attr.dims[2], attr.dims[3], attr.n_elems);
        return 0;
    }

    static bool layout_printed = false;
    if (!layout_printed) {
#if YOLOV5_POSTPROCESS_VERBOSE_LOG
        printf("[YOLOv5] single output parser: n_dims=%d dims=[%d,%d,%d,%d] class_dim=%d proposals=%d flat=%d\n",
               attr.n_dims, attr.dims[0], attr.dims[1], attr.dims[2], attr.dims[3],
               class_dim, proposal_count, flat_layout ? 1 : 0);
#endif
        layout_printed = true;
    }

    int valid_count = 0;
    for (int i = 0; i < proposal_count; ++i) {
        int base_offset = 0;
        if (flat_layout) {
            base_offset = i * box_size;
        } else {
            int remain = i;
            for (int axis_i = proposal_axis_count - 1; axis_i >= 0; --axis_i) {
                int axis = proposal_axes[axis_i];
                int coord = remain % dims[axis];
                remain /= dims[axis];
                base_offset += coord * strides[axis];
            }
        }

        int offset0 = flat_layout ? base_offset : base_offset + 0 * strides[class_dim];
        int offset1 = flat_layout ? base_offset + 1 : base_offset + 1 * strides[class_dim];
        int offset2 = flat_layout ? base_offset + 2 : base_offset + 2 * strides[class_dim];
        int offset3 = flat_layout ? base_offset + 3 : base_offset + 3 * strides[class_dim];
        int offset4 = flat_layout ? base_offset + 4 : base_offset + 4 * strides[class_dim];
        int offset_person = flat_layout ? base_offset + 5 + YOLOV5_PERSON_CLASS_ID
                                        : base_offset + (5 + YOLOV5_PERSON_CLASS_ID) * strides[class_dim];
        if (offset_person >= attr.n_elems) {
            continue;
        }

        float bx = deqnt_output_to_f32(input, offset0, attr);
        float by = deqnt_output_to_f32(input, offset1, attr);
        float bw = deqnt_output_to_f32(input, offset2, attr);
        float bh = deqnt_output_to_f32(input, offset3, attr);
        float box_conf = deqnt_output_to_f32(input, offset4, attr);
        float person_prob = deqnt_output_to_f32(input, offset_person, attr);

        float score = box_conf * person_prob;
        if (score < threshold) continue;

        // Common exported YOLOv5 single output is xywh in model input pixels.
        // If it is normalized, scale it back to model input space.
        if (bw <= 2.f && bh <= 2.f && bx <= 2.f && by <= 2.f) {
            bx *= (float)model_w;
            by *= (float)model_h;
            bw *= (float)model_w;
            bh *= (float)model_h;
        }

        boxes.push_back(bx - bw * 0.5f);
        boxes.push_back(by - bh * 0.5f);
        boxes.push_back(bw);
        boxes.push_back(bh);
        scores.push_back(score);
        valid_count++;
    }

    return valid_count;
}

int post_process(struct yolov5_app_context *app_ctx,
                 rknn_tensor_mem **outputs,
                 float conf_threshold,
                 float nms_threshold,
                 yolov5_person_result_list_t *od_results)
{
    if (!app_ctx || !outputs || !od_results) return -1;

    memset(od_results, 0, sizeof(*od_results));

    std::vector<float> boxes;
    std::vector<float> scores;
    boxes.reserve(128 * 4);
    scores.reserve(128);

    const int model_w = app_ctx->model_width;
    const int model_h = app_ctx->model_height;
    int valid_count = 0;

    if (app_ctx->io_num.n_output == 1) {
        if (!outputs[0]) return -1;
        valid_count += process_person_single_output_quant((uint8_t *)outputs[0]->virt_addr,
                                                          app_ctx->output_attrs[0],
                                                          boxes,
                                                          scores,
                                                          conf_threshold,
                                                          model_w,
                                                          model_h);
    } else if (app_ctx->io_num.n_output == YOLOV5_OUTPUT_BRANCHES) {
        bool branch_quant = true;
        for (int i = 0; i < (int)app_ctx->io_num.n_output; ++i) {
            rknn_tensor_type type = app_ctx->output_attrs[i].type;
            if (type != RKNN_TENSOR_INT8 && type != RKNN_TENSOR_UINT8) {
                branch_quant = false;
                break;
            }
        }
        if (!branch_quant) {
            printf("[YOLOv5] unsupported branch output type for person postprocess\n");
            return -1;
        }

        for (int i = 0; i < (int)app_ctx->io_num.n_output; ++i) {
            if (!outputs[i]) return -1;
            int grid_h = app_ctx->output_attrs[i].dims[2];
            int grid_w = app_ctx->output_attrs[i].dims[1];
            int stride = grid_h > 0 ? model_h / grid_h : 0;
            if (grid_h <= 0 || grid_w <= 0 || stride <= 0) {
                printf("[YOLOv5] invalid output grid branch=%d h=%d w=%d stride=%d\n",
                       i, grid_h, grid_w, stride);
                continue;
            }

            valid_count += process_person_i8_rv1106((int8_t *)outputs[i]->virt_addr,
                                                    kAnchor[i],
                                                    grid_h,
                                                    grid_w,
                                                    stride,
                                                    boxes,
                                                    scores,
                                                    conf_threshold,
                                                    app_ctx->output_attrs[i].zp,
                                                    app_ctx->output_attrs[i].scale);
        }
    } else {
        printf("[YOLOv5] unsupported output count=%u\n", app_ctx->io_num.n_output);
        return -1;
    }

    if (scores.empty()) return 0;

    std::vector<int> order(scores.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = (int)i;

    std::sort(order.begin(), order.end(), [&scores](int a, int b) {
        return scores[a] > scores[b];
    });
    nms_person(boxes, scores, order, nms_threshold);

    for (size_t i = 0; i < order.size() && od_results->count < YOLOV5_MAX_PERSON_RESULTS; ++i) {
        int n = order[i];
        if (n < 0) continue;

        float x1 = boxes[n * 4 + 0];
        float y1 = boxes[n * 4 + 1];
        float x2 = x1 + boxes[n * 4 + 2];
        float y2 = y1 + boxes[n * 4 + 3];

        yolov5_person_result_t *dst = &od_results->results[od_results->count++];
        dst->box.left = clamp_i(x1, 0, model_w);
        dst->box.top = clamp_i(y1, 0, model_h);
        dst->box.right = clamp_i(x2, 0, model_w);
        dst->box.bottom = clamp_i(y2, 0, model_h);
        dst->prop = scores[n];
        dst->cls_id = YOLOV5_PERSON_CLASS_ID;
    }

    return 0;
}

yolov5_on_duty_status_t evaluate_on_duty_status(const yolov5_person_result_list_t *results,
                                                int image_width,
                                                int image_height,
                                                float min_score,
                                                float min_area_ratio)
{
    yolov5_on_duty_status_t status;
    memset(&status, 0, sizeof(status));
    status.best_score = 0.f;

    if (!results || image_width <= 0 || image_height <= 0) {
        return status;
    }

    float image_area = (float)image_width * (float)image_height;
    for (int i = 0; i < results->count; ++i) {
        const yolov5_person_result_t *r = &results->results[i];
        int w = r->box.right - r->box.left;
        int h = r->box.bottom - r->box.top;
        if (w <= 0 || h <= 0) continue;

        float area_ratio = ((float)w * (float)h) / image_area;
        if (r->prop > status.best_score) {
            status.best_score = r->prop;
            status.best_box = r->box;
        }
        if (r->prop >= min_score && area_ratio >= min_area_ratio) {
            status.on_duty = true;
            status.person_count++;
        }
    }

    return status;
}

int init_post_process(void)
{
    return 0;
}

const char *coco_cls_to_name(int cls_id)
{
    return cls_id == YOLOV5_PERSON_CLASS_ID ? "person" : "unused";
}

void deinit_post_process(void)
{
}
