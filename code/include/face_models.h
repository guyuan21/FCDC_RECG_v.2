// =============================================================================
// face_models.h  —  RV1106 Pro Max 人脸考勤/权限管理系统
// Platform : Luckfox Pico Pro Max (RV1106)  |  Display: 2.8" TFT
// =============================================================================
#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include <array>
#include <string>
#include <vector>

#include "rknn_api.h"
// rknn_api.h 已内联定义 get_type_string / get_format_string / get_qnt_type_string
// 本头文件不再重复声明，直接使用 rknn_api.h 提供的版本

#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


// -----------------------------------------------------------------------------
// 平台常量
// -----------------------------------------------------------------------------
#define RETINA_INPUT_W      640
#define RETINA_INPUT_H      640
#define FACENET_INPUT_W     112
#define FACENET_INPUT_H     112
#define FACENET_FEAT_DIM    512
#define MAX_DETECT_FACES    4       // 单帧最多处理人脸数

// 识别阈值：欧氏距离，值越小越严格（L2归一化后范围[0,2]）
// 当前策略：
// 1) best_dist 必须小于 FACE_RECOG_THRESHOLD
// 2) best_dist 与 second_best_dist 必须至少拉开 FACE_RECOG_MARGIN
//    避免“像两个人都很接近”时误识别到错误身份
#define FACE_RECOG_THRESHOLD    0.72f
#define FACE_RECOG_MARGIN       0.10f
#define FACE_CONFIRM_FRAMES     3       // 连续N帧识别相同人才确认

// 考勤业务间隔（秒）：协议模式由上位机驱动，不做本地时间间隔保护。
#define ATTENDANCE_DUPLICATE_COOLDOWN_S  0
#define ATTENDANCE_COOLDOWN_S            ATTENDANCE_DUPLICATE_COOLDOWN_S

// 人脸质量过滤：ROI边长小于此值时跳过识别
#define MIN_FACE_ROI_SIZE       56
#define MIN_FACE_DETECT_SCORE   0.50f
#define FACE_DB_MAX_IMAGE_W     640
#define FACE_DB_MAX_IMAGE_H     480

// -----------------------------------------------------------------------------
// RKNN 上下文
// -----------------------------------------------------------------------------
#define RKNN_MAX_IO 16

typedef struct {
    rknn_context        rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr   *input_attrs;
    rknn_tensor_attr   *output_attrs;
    rknn_tensor_mem    *input_mems[RKNN_MAX_IO];
    rknn_tensor_mem    *output_mems[RKNN_MAX_IO];
    rknn_tensor_mem    *net_mem;     // MEM_SHARE 模式
    rknn_tensor_mem    *max_mem;
    int  model_width;
    int  model_height;
    int  model_channel;
    bool is_quant;
} rknn_app_context_t;

typedef struct {
    int   src_w;
    int   src_h;
    int   dst_w;
    int   dst_h;
    int   resized_w;
    int   resized_h;
    int   pad_left;
    int   pad_top;
    float scale;
} image_transform_t;

// -----------------------------------------------------------------------------
// 检测结果
// -----------------------------------------------------------------------------
typedef struct { int left, top, right, bottom; } detect_box_t;
typedef struct { int x, y; }                     detect_point_t;

typedef struct {
    detect_box_t    box;
    detect_point_t  point[5];   // 五个关键点
    float           prop;       // 置信度
} object_detect_result;

typedef struct {
    object_detect_result results[MAX_DETECT_FACES];
    int count;
} object_detect_result_list;

// -----------------------------------------------------------------------------
// 人脸数据库条目
// -----------------------------------------------------------------------------
struct FaceEntry {
    std::string name;
    int         worker_id;                  // 工号
    bool        has_permission;             // 是否有门禁权限
    std::vector<std::array<float, FACENET_FEAT_DIM>> features; // 每张样本图保留一条特征
    int         sample_count = 0;   // 已录入样本数
};

// -----------------------------------------------------------------------------
// 考勤记录
// -----------------------------------------------------------------------------
struct AttendanceRecord {
    std::string name;
    int         worker_id;
    time_t      timestamp;
    bool        is_checkin;     // true=签入 false=签出
};

// -----------------------------------------------------------------------------
// 系统状态（用于LCD UI渲染）
// -----------------------------------------------------------------------------
enum class SysStatus {
    IDLE,           // 待机
    DETECTING,      // 检测中
    RECOGNIZED,     // 识别成功
    STRANGER,       // 陌生人
    ACCESS_GRANT,   // 打卡通过
    ACCESS_DENY,    // 打卡拒绝
};

// -----------------------------------------------------------------------------
// 函数声明
// -----------------------------------------------------------------------------

// --- 模型初始化/释放 ---
int  init_face_models(const char *detector_path,
                      const char *recognizer_path,
                      rknn_app_context_t *detector_ctx,
                      rknn_app_context_t *recognizer_ctx);
int  release_face_detector(rknn_app_context_t *app_ctx);
int  release_face_recognizer(rknn_app_context_t *app_ctx);

// --- 推理 ---
int  run_face_detector(rknn_app_context_t *app_ctx,
                       object_detect_result_list *od_results);

// --- 特征提取与匹配 ---
// 提取特征到 out_fp32（已L2归一化）
void extract_face_embedding(rknn_app_context_t *recognizer_ctx,
                            const cv::Mat &face_roi,
                            float *out_fp32);

// 欧氏距离（两个L2归一化向量）
float get_euclidean_distance(const float *a, const float *b, int dim = FACENET_FEAT_DIM);

// 余弦相似度（L2归一化后等价于点积）
float get_cosine_similarity(const float *a, const float *b, int dim = FACENET_FEAT_DIM);

// 对feature做L2归一化
void l2_normalize(float *feat, int dim = FACENET_FEAT_DIM);

// dequant int8→float
void output_normalization(rknn_app_context_t *app_ctx,
                          uint8_t *output, float *out_fp32);

// --- 人脸数据库 ---
// db_path 可传入 faces 目录（推荐，子目录格式：name_workerId_perm）或旧版 face_db.txt 文件。
int  load_face_db(const char *db_path,
                  rknn_app_context_t *retina_ctx,
                  rknn_app_context_t *facenet_ctx);

// 返回最佳匹配的FaceEntry指针，未识别返回nullptr；out_dist 同时输出最小距离
const FaceEntry *match_face(const float *query, float *out_dist = nullptr);

// 在线注册新人脸（采集当前帧）
bool register_face(const std::string &name, int worker_id, bool has_permission,
                   rknn_app_context_t *facenet_ctx, const cv::Mat &face_roi,
                   const char *db_path);

// 在线注册新人脸（采集多帧样本，最终合成同一个人）
bool register_face_samples(const std::string &name, int worker_id, bool has_permission,
                           rknn_app_context_t *facenet_ctx,
                           const std::vector<cv::Mat> &face_samples,
                           const char *db_path);
bool remove_face_employee(int worker_id, const char *db_path);

// --- 图像处理 ---
// letterbox缩放（写入到已分配的cv::Mat或raw buffer）
void letterbox_to_mat(const cv::Mat &src, cv::Mat &dst,
                      image_transform_t *transform = nullptr);
void letterbox_to_buf(const cv::Mat &src, unsigned char *dst_buf, int dst_w, int dst_h,
                      image_transform_t *transform = nullptr);

// 坐标映射：将 letterbox 后的模型坐标还原到原图坐标
bool map_coordinates(const image_transform_t &transform, int *x, int *y);
void scale_coordinates(const cv::Size &src_space, const cv::Size &dst_space,
                       int *x, int *y);
bool align_face_from_detection(const cv::Mat &src_bgr,
                               const object_detect_result *det,
                               const image_transform_t *transform,
                               cv::Mat &aligned_face);

// --- 全局数据库访问 ---
std::vector<FaceEntry>& get_face_db();
