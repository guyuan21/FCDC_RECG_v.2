// =============================================================================
// face_recognizer.cc  —  FaceNet 特征提取 + 人脸数据库 + 考勤管理
// Platform : Luckfox Pico Pro Max (RV1106)
// =============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <errno.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <unordered_map>
#include <vector>

#include "face_models.h"

#define FACENET_VERBOSE_LOG 0
#define FACE_FEATURE_CACHE_MAGIC "FCDCFEAT"
#define FACE_FEATURE_CACHE_VERSION 1u

// =============================================================================
// 全局状态
// =============================================================================
static std::vector<FaceEntry>          g_face_db;

static bool face_db_verbose_log_enabled()
{
    static int enabled = -1;
    if (enabled < 0) {
        const char *value = getenv("ATTENDANCE_DB_VERBOSE");
        enabled = (value && strcmp(value, "1") == 0) ? 1 : 0;
    }
    return enabled != 0;
}

struct FaceFeatureCacheHeader {
    char magic[8];
    uint32_t version;
    uint32_t dim;
    uint64_t path_hash;
    uint64_t image_size;
    uint64_t image_mtime;
};

static const cv::Point2f kFaceAlignTemplate[5] = {
    cv::Point2f(38.2946f, 51.6963f),
    cv::Point2f(73.5318f, 51.5014f),
    cv::Point2f(56.0252f, 71.7366f),
    cv::Point2f(41.5493f, 92.3655f),
    cv::Point2f(70.7299f, 92.2041f),
};

// =============================================================================
// 工具函数
// =============================================================================
static bool has_supported_face_ext(const std::string &path)
{
    const char *dot = strrchr(path.c_str(), '.');
    if (!dot) return false;

    std::string ext(dot + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext == "jpg" || ext == "jpeg" || ext == "png");
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

static bool is_directory_path(const std::string &path)
{
    struct stat st;
    return (stat(path.c_str(), &st) == 0) && S_ISDIR(st.st_mode);
}

static std::string join_path(const std::string &base, const std::string &name)
{
    if (name.empty()) return base;
    if (!name.empty() && name[0] == '/') return name;
    if (base.empty() || base == ".") return name;
    return base + "/" + name;
}

static std::string parent_dir_of(const std::string &path)
{
    size_t pos = path.rfind('/');
    if (pos == std::string::npos) return ".";
    if (pos == 0) return "/";
    return path.substr(0, pos);
}

static uint64_t fnv1a64(const std::string &text)
{
    uint64_t hash = 1469598103934665603ull;
    for (unsigned char ch : text) {
        hash ^= (uint64_t)ch;
        hash *= 1099511628211ull;
    }
    return hash;
}

static std::string hex_u64(uint64_t value)
{
    char buf[17];
    snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)value);
    return std::string(buf);
}

static bool ensure_directory_exists(const std::string &path)
{
    if (path.empty()) return false;

    struct stat st;
    if (stat(path.c_str(), &st) == 0) {
        return S_ISDIR(st.st_mode);
    }
    return mkdir(path.c_str(), 0755) == 0;
}

static std::string feature_cache_path(const std::string &cache_dir,
                                      uint64_t path_hash)
{
    if (cache_dir.empty()) return "";
    return join_path(cache_dir, hex_u64(path_hash) + ".feat");
}

static std::string face_db_manifest_path(const char *db_path,
                                         bool *db_is_dir,
                                         std::string *db_dir)
{
    if (db_is_dir) *db_is_dir = false;
    if (db_dir) db_dir->assign(".");
    if (!db_path || !db_path[0]) return "";

    if (is_directory_path(db_path)) {
        if (db_is_dir) *db_is_dir = true;
        if (db_dir) db_dir->assign(db_path);
        return join_path(db_path, "face_db.txt");
    }

    if (db_dir) db_dir->assign(parent_dir_of(db_path));
    return db_path;
}

static std::string sanitize_filename_component(const std::string &text)
{
    std::string out;
    out.reserve(text.size());
    for (unsigned char ch : text) {
        if (ch <= 0x20u || ch == 0x7fu || ch == 0x2fu || ch == 0x5cu || ch == 0x3au) {
            out.push_back('_');
        } else {
            out.push_back((char)ch);
        }
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    if (out.empty()) out = "unknown";
    return out;
}

static std::string face_entry_key(const std::string &name, int worker_id)
{
    return name + "#" + std::to_string(worker_id);
}

static std::string face_person_dir_path(const std::string &db_dir,
                                        const std::string &name,
                                        int worker_id,
                                        bool has_permission)
{
    std::string safe_name = sanitize_filename_component(name);
    char dir_name[256];
    snprintf(dir_name, sizeof(dir_name), "%s_%d_%d",
             safe_name.c_str(), worker_id, has_permission ? 1 : 0);
    return join_path(db_dir, dir_name);
}

static std::string face_snapshot_path(const std::string &person_dir,
                                      size_t sample_index)
{
    char file_name[64];
    snprintf(file_name, sizeof(file_name), "sample_%02zu.png", sample_index + 1u);
    return join_path(person_dir, file_name);
}

static bool remove_path_recursive(const std::string &path)
{
    struct stat st;
    if (path.empty()) return false;
    if (lstat(path.c_str(), &st) != 0) {
        return errno == ENOENT;
    }

    if (!S_ISDIR(st.st_mode)) {
        if (unlink(path.c_str()) != 0 && errno != ENOENT) {
            printf("[DB] Remove file failed: %s (%s)\n",
                   path.c_str(), strerror(errno));
            return false;
        }
        return true;
    }

    DIR *dir = opendir(path.c_str());
    if (!dir) {
        printf("[DB] Open dir failed: %s (%s)\n",
               path.c_str(), strerror(errno));
        return false;
    }

    bool ok = true;
    struct dirent *entry = nullptr;
    while ((entry = readdir(dir)) != nullptr) {
        if (strcmp(entry->d_name, ".") == 0 ||
            strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        if (!remove_path_recursive(join_path(path, entry->d_name))) {
            ok = false;
        }
    }
    closedir(dir);

    if (rmdir(path.c_str()) != 0 && errno != ENOENT) {
        printf("[DB] Remove dir failed: %s (%s)\n",
               path.c_str(), strerror(errno));
        ok = false;
    }
    return ok;
}

static void collect_face_images_from_dir(const std::string &dir_path,
                                         std::vector<std::string> &paths)
{
    DIR *dir = opendir(dir_path.c_str());
    if (!dir) {
        printf("[DB] Cannot open dir: %s\n", dir_path.c_str());
        return;
    }

    std::vector<std::string> file_names;
    struct dirent *entry = nullptr;
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_name[0] == '.') continue;
        std::string full_path = join_path(dir_path, entry->d_name);
        if (is_directory_path(full_path)) continue;
        if (has_supported_face_ext(full_path)) file_names.push_back(full_path);
    }
    closedir(dir);

    std::sort(file_names.begin(), file_names.end());
    paths.insert(paths.end(), file_names.begin(), file_names.end());
}

static void resolve_face_token_paths(const std::string &db_dir,
                                     const std::string &token,
                                     std::vector<std::string> &paths)
{
    if (token.empty() || token == "-") return;

    std::string resolved = join_path(db_dir, token);
    if (is_directory_path(resolved)) {
        collect_face_images_from_dir(resolved, paths);
        return;
    }

    if (!has_supported_face_ext(resolved)) {
        printf("[DB] Skip unsupported token: %s\n", resolved.c_str());
        return;
    }
    paths.push_back(resolved);
}

static bool parse_int_strict(const std::string &text, int *value)
{
    if (!value || text.empty()) return false;
    for (char ch : text) {
        if (!isdigit((unsigned char)ch)) return false;
    }
    *value = atoi(text.c_str());
    return true;
}

static bool parse_face_folder_name(const std::string &folder_name,
                                   std::string &name,
                                   int *worker_id,
                                   int *permission)
{
    size_t perm_sep = folder_name.rfind('_');
    if (perm_sep == std::string::npos || perm_sep + 1 >= folder_name.size()) return false;

    size_t id_sep = folder_name.rfind('_', perm_sep - 1);
    if (id_sep == std::string::npos || id_sep == 0 || id_sep + 1 >= perm_sep) return false;

    int parsed_id = 0;
    int parsed_perm = 0;
    if (!parse_int_strict(folder_name.substr(id_sep + 1, perm_sep - id_sep - 1), &parsed_id) ||
        !parse_int_strict(folder_name.substr(perm_sep + 1), &parsed_perm)) {
        return false;
    }

    name = folder_name.substr(0, id_sep);
    if (name.empty()) return false;

    if (worker_id) *worker_id = parsed_id;
    if (permission) *permission = parsed_perm;
    return true;
}

static bool decode_face_image(const std::string &img_path, cv::Mat &img_bgr)
{
    if (access(img_path.c_str(), R_OK) != 0) {
        img_bgr.release();
        return false;
    }

    img_bgr = cv::imread(img_path, cv::IMREAD_COLOR);
    if (img_bgr.empty()) {
        printf("[DB] Skip (decode failed): %s\n", img_path.c_str());
        return false;
    }

    const bool landscape = img_bgr.cols >= img_bgr.rows;
    const int max_w = landscape ? FACE_DB_MAX_IMAGE_W : FACE_DB_MAX_IMAGE_H;
    const int max_h = landscape ? FACE_DB_MAX_IMAGE_H : FACE_DB_MAX_IMAGE_W;
    if (img_bgr.cols > max_w || img_bgr.rows > max_h) {
        const float scale = std::min((float)max_w / img_bgr.cols,
                                     (float)max_h / img_bgr.rows);
        cv::resize(img_bgr, img_bgr, cv::Size(), scale, scale, cv::INTER_AREA);
    }
    return true;
}

static bool read_face_feature_cache(const std::string &cache_dir,
                                    const std::string &img_path,
                                    float *feat)
{
    if (cache_dir.empty() || !feat) return false;

    struct stat st;
    if (stat(img_path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) {
        return false;
    }

    uint64_t path_hash = fnv1a64(img_path);
    std::string cache_path = feature_cache_path(cache_dir, path_hash);
    FILE *fp = fopen(cache_path.c_str(), "rb");
    if (!fp) return false;

    FaceFeatureCacheHeader header;
    bool ok = fread(&header, sizeof(header), 1, fp) == 1;
    ok = ok &&
         memcmp(header.magic, FACE_FEATURE_CACHE_MAGIC, sizeof(header.magic)) == 0 &&
         header.version == FACE_FEATURE_CACHE_VERSION &&
         header.dim == FACENET_FEAT_DIM &&
         header.path_hash == path_hash &&
         header.image_size == (uint64_t)st.st_size &&
         header.image_mtime == (uint64_t)st.st_mtime;
    if (ok) {
        ok = fread(feat, sizeof(float), FACENET_FEAT_DIM, fp) == FACENET_FEAT_DIM;
    }
    fclose(fp);
    return ok;
}

static void write_face_feature_cache(const std::string &cache_dir,
                                     const std::string &img_path,
                                     const float *feat)
{
    if (cache_dir.empty() || !feat) return;

    struct stat st;
    if (stat(img_path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) {
        return;
    }
    if (!ensure_directory_exists(cache_dir)) return;

    uint64_t path_hash = fnv1a64(img_path);
    std::string cache_path = feature_cache_path(cache_dir, path_hash);
    FILE *fp = fopen(cache_path.c_str(), "wb");
    if (!fp) return;

    FaceFeatureCacheHeader header;
    memset(&header, 0, sizeof(header));
    memcpy(header.magic, FACE_FEATURE_CACHE_MAGIC, sizeof(header.magic));
    header.version = FACE_FEATURE_CACHE_VERSION;
    header.dim = FACENET_FEAT_DIM;
    header.path_hash = path_hash;
    header.image_size = (uint64_t)st.st_size;
    header.image_mtime = (uint64_t)st.st_mtime;

    fwrite(&header, sizeof(header), 1, fp);
    fwrite(feat, sizeof(float), FACENET_FEAT_DIM, fp);
    fclose(fp);
}

static void append_face_feature(FaceEntry &entry, const float *feat)
{
    std::array<float, FACENET_FEAT_DIM> sample;
    for (int i = 0; i < FACENET_FEAT_DIM; ++i) {
        sample[i] = feat[i];
    }
    l2_normalize(sample.data());
    entry.features.push_back(sample);
    entry.sample_count = (int)entry.features.size();
}

static bool map_detection_to_source(const object_detect_result *det,
                                    const image_transform_t *transform,
                                    object_detect_result *mapped_det)
{
    if (!det || !mapped_det) return false;
    *mapped_det = *det;

    if (!transform) return true;

    if (!map_coordinates(*transform, &mapped_det->box.left,  &mapped_det->box.top) ||
        !map_coordinates(*transform, &mapped_det->box.right, &mapped_det->box.bottom)) {
        return false;
    }

    for (int i = 0; i < 5; ++i) {
        if (!map_coordinates(*transform, &mapped_det->point[i].x, &mapped_det->point[i].y)) {
            return false;
        }
    }
    return true;
}

static bool crop_face_from_detection(const cv::Mat &src_bgr,
                                     const object_detect_result &det,
                                     cv::Mat &face_roi)
{
    if (src_bgr.empty()) return false;

    int left   = std::max(0, det.box.left);
    int top    = std::max(0, det.box.top);
    int right  = std::min(src_bgr.cols - 1, det.box.right);
    int bottom = std::min(src_bgr.rows - 1, det.box.bottom);
    int width  = right - left;
    int height = bottom - top;
    if (width < MIN_FACE_ROI_SIZE || height < MIN_FACE_ROI_SIZE) return false;

    face_roi = src_bgr(cv::Rect(left, top, width, height)).clone();
    return !face_roi.empty();
}

bool align_face_from_detection(const cv::Mat &src_bgr,
                               const object_detect_result *det,
                               const image_transform_t *transform,
                               cv::Mat &aligned_face)
{
    aligned_face.release();
    if (src_bgr.empty() || !det) return false;

    object_detect_result mapped_det;
    if (!map_detection_to_source(det, transform, &mapped_det)) return false;

    cv::Point2f src_points[5];
    cv::Point2f src_mean(0.f, 0.f);
    cv::Point2f dst_mean(0.f, 0.f);
    for (int i = 0; i < 5; ++i) {
        src_points[i] = cv::Point2f((float)mapped_det.point[i].x,
                                    (float)mapped_det.point[i].y);
        src_mean += src_points[i];
        dst_mean += kFaceAlignTemplate[i];
    }
    src_mean *= 1.f / 5.f;
    dst_mean *= 1.f / 5.f;

    float den = 0.f;
    float a_num = 0.f;
    float b_num = 0.f;
    for (int i = 0; i < 5; ++i) {
        float x = src_points[i].x - src_mean.x;
        float y = src_points[i].y - src_mean.y;
        float u = kFaceAlignTemplate[i].x - dst_mean.x;
        float v = kFaceAlignTemplate[i].y - dst_mean.y;
        den += x * x + y * y;
        a_num += x * u + y * v;
        b_num += x * v - y * u;
    }

    if (den < 1e-6f) {
        return crop_face_from_detection(src_bgr, mapped_det, aligned_face);
    }

    float a = a_num / den;
    float b = b_num / den;
    float tx = dst_mean.x - a * src_mean.x + b * src_mean.y;
    float ty = dst_mean.y - b * src_mean.x - a * src_mean.y;

    cv::Matx23f affine(a, -b, tx,
                       b,  a, ty);
    cv::warpAffine(src_bgr, aligned_face, affine,
                   cv::Size(FACENET_INPUT_W, FACENET_INPUT_H),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    if (aligned_face.empty()) {
        return crop_face_from_detection(src_bgr, mapped_det, aligned_face);
    }
    return true;
}

static bool detect_and_align_face(rknn_app_context_t *retina_ctx,
                                  const cv::Mat &img_bgr,
                                  cv::Mat &aligned_face)
{
    aligned_face.release();
    if (!retina_ctx || img_bgr.empty()) return false;

    cv::Mat retina_input(RETINA_INPUT_H, RETINA_INPUT_W, CV_8UC3,
                         retina_ctx->input_mems[0]->virt_addr);
    image_transform_t transform;
    memset(&transform, 0, sizeof(transform));
    letterbox_to_mat(img_bgr, retina_input, &transform);

    object_detect_result_list od_results;
    memset(&od_results, 0, sizeof(od_results));
    if (run_face_detector(retina_ctx, &od_results) != 0 || od_results.count <= 0) {
        return false;
    }

    int best_idx = 0;
    float best_score = -1.f;
    for (int i = 0; i < od_results.count; ++i) {
        const object_detect_result &det = od_results.results[i];
        if (det.prop < MIN_FACE_DETECT_SCORE) continue;
        const float area = (float)(det.box.right - det.box.left) * (det.box.bottom - det.box.top);
        const float score = det.prop + area * 1e-5f;
        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }
    if (best_score < 0.f) return false;

    return align_face_from_detection(img_bgr, &od_results.results[best_idx], &transform, aligned_face);
}

// =============================================================================
// 模型初始化（单模型，供独立使用）
// =============================================================================
int init_facenet_model(const char *model_path, rknn_app_context_t *app_ctx)
{
    if (!model_path || !app_ctx) return -1;
    memset(app_ctx, 0, sizeof(*app_ctx));

    rknn_context ctx = 0;
    int ret = rknn_init(&ctx, (char *)model_path, 0, 0, NULL);
    if (ret < 0) { printf("[FaceNet] rknn_init fail ret=%d\n", ret); return -1; }
    app_ctx->rknn_ctx = ctx;

    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("[FaceNet] query io_num fail\n");
        release_face_recognizer(app_ctx);
        return -1;
    }
    if (io_num.n_input > RKNN_MAX_IO || io_num.n_output > RKNN_MAX_IO ||
        io_num.n_input == 0 || io_num.n_output == 0) {
        printf("[FaceNet] unsupported io count input=%u output=%u\n",
               io_num.n_input, io_num.n_output);
        release_face_recognizer(app_ctx);
        return -1;
    }
    app_ctx->io_num = io_num;

    // --- 输入属性 ---
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (uint32_t i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &input_attrs[i], sizeof(rknn_tensor_attr));
    }

    // --- 输出属性 ---
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (uint32_t i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr));
    }

    // --- Zero-copy 输入内存（UINT8 NHWC）---
    input_attrs[0].type = RKNN_TENSOR_UINT8;
    input_attrs[0].fmt  = RKNN_TENSOR_NHWC;
    app_ctx->input_mems[0] = rknn_create_mem(ctx, input_attrs[0].size_with_stride);
    if (!app_ctx->input_mems[0]) {
        printf("[FaceNet] create input mem fail\n");
        release_face_recognizer(app_ctx);
        return -1;
    }
    ret = rknn_set_io_mem(ctx, app_ctx->input_mems[0], &input_attrs[0]);
    if (ret < 0) {
        printf("[FaceNet] set input mem fail\n");
        release_face_recognizer(app_ctx);
        return -1;
    }

    // --- Zero-copy 输出内存 ---
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        app_ctx->output_mems[i] = rknn_create_mem(ctx, output_attrs[i].size_with_stride);
        if (!app_ctx->output_mems[i]) {
            printf("[FaceNet] create output mem[%d] fail\n", i);
            release_face_recognizer(app_ctx);
            return -1;
        }
        ret = rknn_set_io_mem(ctx, app_ctx->output_mems[i], &output_attrs[i]);
        if (ret < 0) {
            printf("[FaceNet] set output mem[%d] fail\n", i);
            release_face_recognizer(app_ctx);
            return -1;
        }
    }

    app_ctx->is_quant  = (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC);

    app_ctx->input_attrs  = (rknn_tensor_attr *)malloc(io_num.n_input  * sizeof(rknn_tensor_attr));
    app_ctx->output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    if (!app_ctx->input_attrs || !app_ctx->output_attrs) {
        printf("[FaceNet] alloc tensor attrs fail\n");
        release_face_recognizer(app_ctx);
        return -1;
    }
    memcpy(app_ctx->input_attrs,  input_attrs,  io_num.n_input  * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    parse_input_dims(input_attrs[0],
                     &app_ctx->model_height,
                     &app_ctx->model_width,
                     &app_ctx->model_channel);

#if FACENET_VERBOSE_LOG
    printf("[FaceNet] ready  H=%d W=%d C=%d  quant=%d\n",
           app_ctx->model_height, app_ctx->model_width,
           app_ctx->model_channel, app_ctx->is_quant);
#endif
    return 0;
}

// =============================================================================
// 释放模型资源
// =============================================================================
int release_face_recognizer(rknn_app_context_t *app_ctx)
{
    // 先销毁 IO 内存，再销毁上下文
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
    free(app_ctx->input_attrs);  app_ctx->input_attrs  = nullptr;
    free(app_ctx->output_attrs); app_ctx->output_attrs = nullptr;

    if (app_ctx->rknn_ctx) {
        rknn_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    return 0;
}

// =============================================================================
// 图像预处理：letterbox（保持宽高比，填黑边）
// =============================================================================
void letterbox_to_mat(const cv::Mat &src, cv::Mat &dst, image_transform_t *transform)
{
    if (src.empty() || dst.empty()) return;

    int dw = dst.cols, dh = dst.rows;
    float scale = std::min((float)dw / src.cols, (float)dh / src.rows);
    int rw = (int)(src.cols * scale);
    int rh = (int)(src.rows * scale);
    int lpad = (dw - rw) / 2;
    int tpad = (dh - rh) / 2;

    // 清零（黑色填充）
    dst.setTo(cv::Scalar(0, 0, 0));
    cv::Mat roi = dst(cv::Rect(lpad, tpad, rw, rh));
    cv::resize(src, roi, cv::Size(rw, rh), 0, 0, cv::INTER_LINEAR);
    for (int y = 0; y < roi.rows; ++y) {
        cv::Vec3b *row = roi.ptr<cv::Vec3b>(y);
        for (int x = 0; x < roi.cols; ++x) {
            std::swap(row[x][0], row[x][2]);
        }
    }

    if (transform) {
        transform->src_w = src.cols;
        transform->src_h = src.rows;
        transform->dst_w = dst.cols;
        transform->dst_h = dst.rows;
        transform->resized_w = rw;
        transform->resized_h = rh;
        transform->pad_left = lpad;
        transform->pad_top = tpad;
        transform->scale = scale;
    }
}

void letterbox_to_buf(const cv::Mat &src, unsigned char *dst_buf, int dst_w, int dst_h,
                      image_transform_t *transform)
{
    cv::Mat wrapper(dst_h, dst_w, CV_8UC3, dst_buf);
    letterbox_to_mat(src, wrapper, transform);
}

// =============================================================================
// 反量化：int8 → float
// =============================================================================
static inline float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale)
{
    return ((float)qnt - (float)zp) * scale;
}

// =============================================================================
// 输出层 dequant + L2 归一化 → out_fp32[128]
// =============================================================================
void output_normalization(rknn_app_context_t *app_ctx,
                          uint8_t *output, float *out_fp32)
{
    int32_t zp    = app_ctx->output_attrs[0].zp;
    float   scale = app_ctx->output_attrs[0].scale;

    for (int i = 0; i < FACENET_FEAT_DIM; i++)
        out_fp32[i] = deqnt_affine_to_f32((int8_t)output[i], zp, scale);

    l2_normalize(out_fp32);
}

// =============================================================================
// L2 归一化（原地）
// =============================================================================
void l2_normalize(float *feat, int dim)
{
    float sum = 0.f;
    for (int i = 0; i < dim; i++) sum += feat[i] * feat[i];
    float norm = sqrtf(sum);
    if (norm < 1e-7f) return;
    for (int i = 0; i < dim; i++) feat[i] /= norm;
}

// =============================================================================
// 欧氏距离（两向量均已L2归一化时等价于 sqrt(2-2*cos_sim)）
// =============================================================================
float get_euclidean_distance(const float *a, const float *b, int dim)
{
    float sum = 0.f;
    for (int i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sqrtf(sum);
}

static float get_squared_euclidean_distance(const float *a,
                                            const float *b,
                                            int dim,
                                            float stop_at = 3.402823466e+38F)
{
    float sum = 0.f;
    for (int i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        sum += d * d;
        if (sum >= stop_at) break;
    }
    return sum;
}

// =============================================================================
// 余弦相似度（L2归一化后直接点积）—— 相似度越高值越接近1
// =============================================================================
float get_cosine_similarity(const float *a, const float *b, int dim)
{
    float dot = 0.f;
    for (int i = 0; i < dim; i++) dot += a[i] * b[i];
    return dot;  // 已归一化，无需除以模长
}

// =============================================================================
// 特征提取（封装推理流程）
// =============================================================================
void extract_face_embedding(rknn_app_context_t *facenet_ctx,
                           const cv::Mat &face_roi, float *out_fp32)
{
    int h = facenet_ctx->model_height;
    int w = facenet_ctx->model_width;
    cv::Mat input(h, w, CV_8UC3, facenet_ctx->input_mems[0]->virt_addr);
    letterbox_to_mat(face_roi, input);

    if (rknn_run(facenet_ctx->rknn_ctx, nullptr) < 0) {
        printf("[FaceNet] rknn_run failed\n");
        memset(out_fp32, 0, FACENET_FEAT_DIM * sizeof(float));
        return;
    }
    uint8_t *raw = (uint8_t *)(facenet_ctx->output_mems[0]->virt_addr);
    output_normalization(facenet_ctx, raw, out_fp32);
}

// =============================================================================
// 人脸数据库加载
// 支持两种输入：
// 1) 目录：<faces>/<name>_<worker_id>_<permission>/ 图片
// 2) 文件：<name> <worker_id> <permission(0/1)> <img_path1> [img_path2 ...]
// 多张图像的特征向量取均值后重新L2归一化，提升泛化性
// =============================================================================
int load_face_db(const char *db_path,
                 rknn_app_context_t *retina_ctx,
                 rknn_app_context_t *facenet_ctx)
{
    g_face_db.clear();

    bool db_is_dir = false;
    std::string db_dir = ".";
    std::string manifest_path = face_db_manifest_path(db_path, &db_is_dir, &db_dir);
    FILE *fp = nullptr;
    if (!db_is_dir) {
        fp = fopen(manifest_path.c_str(), "r");
        if (!fp) {
            printf("[DB] Cannot open %s\n", manifest_path.c_str());
            return 0;
        }
    }

    char line[1024];
    std::unordered_map<std::string, size_t> face_index;
    std::vector<std::string> image_paths;
    std::string feature_cache_dir = join_path(db_dir, ".face_feature_cache");
    int cache_hit_count = 0;
    int cache_write_count = 0;
    int missing_sample_count = 0;
    int decode_failed_count = 0;
    int align_failed_count = 0;

    std::vector<std::string> db_lines;
    if (db_is_dir) {
        DIR *dir = opendir(db_path);
        if (!dir) {
            printf("[DB] Cannot open face dir: %s\n", db_path);
            return 0;
        }

        struct dirent *entry = nullptr;
        while ((entry = readdir(dir)) != nullptr) {
            if (entry->d_name[0] == '.') continue;

            std::string folder_name = entry->d_name;
            std::string person_dir = join_path(db_path, folder_name);
            if (!is_directory_path(person_dir)) continue;

            std::string name;
            int worker_id = 0;
            int permission = 0;
            if (!parse_face_folder_name(folder_name, name, &worker_id, &permission)) {
                printf("[DB] Skip folder (expect name_id_perm): %s\n", folder_name.c_str());
                continue;
            }

            char spec[1024];
            snprintf(spec, sizeof(spec), "%s %d %d %s",
                     name.c_str(), worker_id, permission, person_dir.c_str());
            db_lines.push_back(spec);
        }
        closedir(dir);
        std::sort(db_lines.begin(), db_lines.end());

        if (access(manifest_path.c_str(), R_OK) == 0) {
            printf("[DB] Directory mode: ignore legacy manifest: %s\n",
                   manifest_path.c_str());
        }
        if (db_lines.empty()) {
            printf("[DB] No employee dirs under: %s\n", db_path);
        }
    } else {
        while (fgets(line, sizeof(line), fp)) {
            if (line[0] == '#' || line[0] == '\n') continue;
            db_lines.push_back(line);
        }
        fclose(fp);
    }

    for (const auto &db_line : db_lines) {
        char line_copy[1024];
        strncpy(line_copy, db_line.c_str(), sizeof(line_copy) - 1);
        line_copy[sizeof(line_copy) - 1] = '\0';
        
        char name[64];
        int worker_id = 0;
        int permission = 0;
        
        // 解析基本信息
        char *ctx_ptr = nullptr;
        char *tok = strtok_r(line_copy, " \t\n\r", &ctx_ptr);
        if (!tok) continue;
        strncpy(name, tok, sizeof(name) - 1);
        name[sizeof(name) - 1] = '\0';
        
        tok = strtok_r(nullptr, " \t\n\r", &ctx_ptr);
        if (tok) worker_id = atoi(tok);
        
        tok = strtok_r(nullptr, " \t\n\r", &ctx_ptr);
        if (tok) permission = atoi(tok);

        image_paths.clear();
        while ((tok = strtok_r(nullptr, " \t\n\r", &ctx_ptr)) != nullptr) {
            resolve_face_token_paths(db_is_dir ? "." : db_dir, tok, image_paths);
        }

        if (image_paths.empty()) {
            printf("[DB] Skip %-16s no valid jpg/png samples\n", name);
            continue;
        }

        FaceEntry *entry = nullptr;
        std::string entry_key = face_entry_key(name, worker_id);
        auto it = face_index.find(entry_key);
        if (it == face_index.end()) {
            FaceEntry new_entry;
            new_entry.name = name;
            new_entry.worker_id = worker_id;
            new_entry.has_permission = (permission != 0);
            new_entry.sample_count = 0;
            g_face_db.push_back(new_entry);
            face_index[entry_key] = g_face_db.size() - 1;
            entry = &g_face_db.back();
        } else {
            entry = &g_face_db[it->second];
        }

        cv::Mat img_bgr;
        cv::Mat aligned_face;
        int loaded_count_before = entry->sample_count;
        for (const auto &img_path : image_paths) {
            float feat[FACENET_FEAT_DIM];
            if (read_face_feature_cache(feature_cache_dir, img_path, feat)) {
                append_face_feature(*entry, feat);
                ++cache_hit_count;
                continue;
            }

            if (access(img_path.c_str(), R_OK) != 0) {
                ++missing_sample_count;
                continue;
            }

            if (!decode_face_image(img_path, img_bgr)) {
                ++decode_failed_count;
                continue;
            }

            if (!detect_and_align_face(retina_ctx, img_bgr, aligned_face)) {
                if (img_bgr.cols >= MIN_FACE_ROI_SIZE && img_bgr.rows >= MIN_FACE_ROI_SIZE) {
                    aligned_face = img_bgr;
                    if (face_db_verbose_log_enabled()) {
                        printf("[DB] Use cropped face sample: %s\n", img_path.c_str());
                    }
                } else {
                    if (face_db_verbose_log_enabled()) {
                        printf("[DB] Skip (no aligned face): %s\n", img_path.c_str());
                    }
                    ++align_failed_count;
                    img_bgr.release();
                    continue;
                }
            }

            extract_face_embedding(facenet_ctx, aligned_face, feat);
            aligned_face.release();
            append_face_feature(*entry, feat);
            write_face_feature_cache(feature_cache_dir, img_path, feat);
            ++cache_write_count;
            img_bgr.release();
        }

        if (entry->sample_count == loaded_count_before) {
            if (loaded_count_before == 0) {
                g_face_db.pop_back();
                face_index.erase(entry_key);
            }
            printf("[DB] Skip %-16s all samples failed\n", name);
            continue;
        }
    }

    for (auto it = g_face_db.begin(); it != g_face_db.end();) {
        if (it->sample_count <= 0) {
            it = g_face_db.erase(it);
            continue;
        }
        if (face_db_verbose_log_enabled()) {
            printf("[DB] Loaded  %-16s  id=%-4d  perm=%d  samples=%d\n",
                   it->name.c_str(), it->worker_id, (int)it->has_permission, it->sample_count);
        }
        ++it;
    }

    if (face_db_verbose_log_enabled() ||
        missing_sample_count > 0 ||
        decode_failed_count > 0 ||
        align_failed_count > 0) {
        printf("[DB] Feature cache hits=%d built=%d missing=%d decode_failed=%d align_failed=%d\n",
               cache_hit_count, cache_write_count, missing_sample_count,
               decode_failed_count, align_failed_count);
    }
    printf("[DB] Total entries: %d\n", (int)g_face_db.size());
    return (int)g_face_db.size();
}

// =============================================================================
// 人脸匹配
// 策略：余弦相似度最大者（等价于L2距离最小）
// 同时要求距离 < FACE_RECOG_THRESHOLD
// =============================================================================
const FaceEntry *match_face(const float *query, float *out_dist)
{
    if (g_face_db.empty()) {
        if (out_dist) *out_dist = -1.f;
        return nullptr;
    }

    float best_person_dist = 9999.f;
    float second_person_dist = 9999.f;
    const FaceEntry *best = nullptr;

    for (const auto &e : g_face_db) {
        float person_dist_sq = 9999.f * 9999.f;
        for (const auto &sample : e.features) {
            float d_sq = get_squared_euclidean_distance(query, sample.data(),
                                                        FACENET_FEAT_DIM,
                                                        person_dist_sq);
            if (d_sq < person_dist_sq) person_dist_sq = d_sq;
        }
        float person_dist = sqrtf(person_dist_sq);

        if (person_dist < best_person_dist) {
            second_person_dist = best_person_dist;
            best_person_dist = person_dist;
            best      = &e;
        } else if (person_dist < second_person_dist) {
            second_person_dist = person_dist;
        }
    }
    if (out_dist) *out_dist = best_person_dist;

    if (!best || best_person_dist >= FACE_RECOG_THRESHOLD) return nullptr;

    if (second_person_dist < 9999.f &&
        (second_person_dist - best_person_dist) < FACE_RECOG_MARGIN) {
        return nullptr;
    }

    return best;
}

// =============================================================================
// 在线注册新人脸
// =============================================================================
bool register_face_samples(const std::string &name, int worker_id, bool has_permission,
                           rknn_app_context_t *facenet_ctx,
                           const std::vector<cv::Mat> &face_samples,
                           const char *db_path)
{
    if (worker_id <= 0) return false;
    if (face_samples.empty()) return false;
    for (const auto &entry : g_face_db) {
        if (entry.worker_id == worker_id) {
            printf("[Reg] Duplicate employee id=%d name=%s\n",
                   worker_id, entry.name.c_str());
            return false;
        }
    }

    std::vector<std::string> snapshot_paths;
    std::string feature_cache_dir;
    std::string person_dir;
    auto cleanup_snapshots = [&]() {
        for (const auto &path : snapshot_paths) {
            unlink(path.c_str());
        }
        if (!person_dir.empty()) rmdir(person_dir.c_str());
    };
    if (db_path && db_path[0]) {
        bool db_is_dir = false;
        std::string db_dir = ".";
        std::string manifest_path = face_db_manifest_path(db_path, &db_is_dir, &db_dir);
        feature_cache_dir = join_path(db_dir, ".face_feature_cache");
        person_dir = face_person_dir_path(db_dir, name, worker_id, has_permission);
        if (!ensure_directory_exists(person_dir)) {
            printf("[Reg] Create person dir failed: %s\n", person_dir.c_str());
            return false;
        }
        snapshot_paths.reserve(face_samples.size());
        for (size_t i = 0; i < face_samples.size(); ++i) {
            const cv::Mat &face_roi = face_samples[i];
            if (face_roi.empty()) {
                continue;
            }
            std::string snapshot_path = face_snapshot_path(person_dir, i);
            if (!cv::imwrite(snapshot_path, face_roi)) {
                printf("[Reg] Save snapshot failed: %s\n", snapshot_path.c_str());
                cleanup_snapshots();
                return false;
            }
            snapshot_paths.push_back(snapshot_path);
        }
        if (snapshot_paths.empty()) {
            printf("[Reg] No valid face samples for %s\n", name.c_str());
            cleanup_snapshots();
            return false;
        }

        if (!db_is_dir) {
            FILE *fp = fopen(manifest_path.c_str(), "a");
            if (!fp) {
                printf("[Reg] Open manifest failed: %s\n", manifest_path.c_str());
                cleanup_snapshots();
                return false;
            }
            fprintf(fp, "%s %d %d %s\n",
                    name.c_str(), worker_id, (int)has_permission, person_dir.c_str());
            fclose(fp);
        }
    }

    float feat[FACENET_FEAT_DIM];
    FaceEntry entry;
    entry.name           = name;
    entry.worker_id      = worker_id;
    entry.has_permission = has_permission;
    size_t appended_samples = 0;
    size_t snapshot_index = 0;
    for (const auto &face_roi : face_samples) {
        if (face_roi.empty()) continue;
        extract_face_embedding(facenet_ctx, face_roi, feat);
        append_face_feature(entry, feat);
        if (snapshot_index < snapshot_paths.size()) {
            write_face_feature_cache(feature_cache_dir, snapshot_paths[snapshot_index], feat);
        }
        ++snapshot_index;
        ++appended_samples;
    }
    if (appended_samples == 0) {
        cleanup_snapshots();
        return false;
    }
    g_face_db.push_back(entry);
    printf("[Reg] New entry: %s id=%d perm=%d samples=%zu\n",
           name.c_str(), worker_id, (int)has_permission, appended_samples);
    return true;
}

bool register_face(const std::string &name, int worker_id, bool has_permission,
                   rknn_app_context_t *facenet_ctx, const cv::Mat &face_roi,
                   const char *db_path)
{
    std::vector<cv::Mat> face_samples;
    if (!face_roi.empty()) face_samples.push_back(face_roi);
    return register_face_samples(name, worker_id, has_permission,
                                 facenet_ctx, face_samples, db_path);
}

bool remove_face_employee(int worker_id, const char *db_path)
{
    if (worker_id <= 0) return false;

    auto it = std::find_if(g_face_db.begin(), g_face_db.end(),
                           [worker_id](const FaceEntry &entry) {
                               return entry.worker_id == worker_id;
                           });
    if (it == g_face_db.end()) {
        printf("[DB] Remove employee failed, id=%d not found\n", worker_id);
        return false;
    }

    std::string name = it->name;
    bool has_permission = it->has_permission;
    bool storage_ok = true;

    if (db_path && db_path[0]) {
        bool db_is_dir = false;
        std::string db_dir = ".";
        std::string manifest_path = face_db_manifest_path(db_path, &db_is_dir, &db_dir);
        std::string person_dir = face_person_dir_path(db_dir, name, worker_id, has_permission);

        if (!db_is_dir && !manifest_path.empty()) {
            std::vector<std::string> kept_lines;
            FILE *fp = fopen(manifest_path.c_str(), "r");
            if (fp) {
                char line[1024];
                while (fgets(line, sizeof(line), fp)) {
                    char parsed_name[256] = {0};
                    char parsed_path[512] = {0};
                    int parsed_id = 0;
                    int parsed_perm = 0;
                    int fields = sscanf(line, "%255s %d %d %511s",
                                        parsed_name, &parsed_id, &parsed_perm, parsed_path);
                    if (fields >= 3 && parsed_id == worker_id) {
                        if (fields >= 4) person_dir = join_path(db_dir, parsed_path);
                        continue;
                    }
                    kept_lines.emplace_back(line);
                }
                fclose(fp);

                fp = fopen(manifest_path.c_str(), "w");
                if (!fp) {
                    printf("[DB] Rewrite manifest failed: %s\n", manifest_path.c_str());
                    storage_ok = false;
                } else {
                    for (const auto &line : kept_lines) {
                        fputs(line.c_str(), fp);
                    }
                    fclose(fp);
                }
            }
        }

        if (!person_dir.empty() && access(person_dir.c_str(), F_OK) == 0) {
            if (!remove_path_recursive(person_dir)) {
                printf("[DB] Remove person dir failed: %s\n", person_dir.c_str());
                storage_ok = false;
            }
        }
    }

    g_face_db.erase(it);
    printf("[DB] Removed employee face id=%d name=%s storage=%s\n",
           worker_id, name.c_str(), storage_ok ? "ok" : "partial");
    return storage_ok;
}

std::vector<FaceEntry>& get_face_db()
{
    return g_face_db;
}
