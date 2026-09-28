// Lightweight person-only YOLOv5 postprocess for RV1106/RV1103.

#ifndef _RKNN_DEMO_YOLOV5_POSTPROCESS_H_
#define _RKNN_DEMO_YOLOV5_POSTPROCESS_H_

#include <stdbool.h>

#include "common.h"
#include "image_utils.h"
#include "rknn_api.h"

#define YOLOV5_CLASS_NUM          80
#define YOLOV5_PERSON_CLASS_ID    0
#define YOLOV5_PROP_BOX_SIZE      (5 + YOLOV5_CLASS_NUM)
#define YOLOV5_OUTPUT_BRANCHES    3
#define YOLOV5_MAX_OUTPUTS        4
#define YOLOV5_MAX_PERSON_RESULTS 16

#define YOLOV5_BOX_THRESH         0.25f
#define YOLOV5_NMS_THRESH         0.45f
#define YOLOV5_ON_DUTY_MIN_SCORE  0.35f
#define YOLOV5_ON_DUTY_MIN_AREA   0.015f

typedef struct {
    image_rect_t box;
    float prop;
    int cls_id;
} yolov5_person_result_t;

typedef struct {
    int count;
    yolov5_person_result_t results[YOLOV5_MAX_PERSON_RESULTS];
} yolov5_person_result_list_t;

typedef struct {
    bool on_duty;
    int person_count;
    float best_score;
    image_rect_t best_box;
} yolov5_on_duty_status_t;

struct yolov5_app_context;

int post_process(struct yolov5_app_context *app_ctx,
                 rknn_tensor_mem **outputs,
                 float conf_threshold,
                 float nms_threshold,
                 yolov5_person_result_list_t *od_results);

yolov5_on_duty_status_t evaluate_on_duty_status(const yolov5_person_result_list_t *results,
                                                int image_width,
                                                int image_height,
                                                float min_score,
                                                float min_area_ratio);

const char *coco_cls_to_name(int cls_id);
int init_post_process(void);
void deinit_post_process(void);

#endif // _RKNN_DEMO_YOLOV5_POSTPROCESS_H_
