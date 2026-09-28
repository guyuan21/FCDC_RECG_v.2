#include "AS608.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage:\n"
            "  %s [dev] [baud] info\n"
            "  %s [dev] [baud] search\n"
            "  %s [dev] [baud] enroll <id>\n"
            "  %s [dev] [baud] delete <id>\n"
            "  %s [dev] [baud] empty\n",
            prog, prog, prog, prog, prog);
}

int main(int argc, char **argv)
{
    const char *dev_path = "/dev/ttyS1";
    int baudrate = 57600;
    int cmd_index = 1;

    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    if (argc >= 4 && argv[1][0] == '/') {
        dev_path = argv[1];
        baudrate = atoi(argv[2]);
        cmd_index = 3;
    }

    const char *cmd = argv[cmd_index];
    AS608 sensor;
    if (!sensor.openDevice(dev_path, baudrate)) {
        return 1;
    }

    if (strcmp(cmd, "info") == 0) {
        AS608::SysParam param;
        uint8_t ensure = sensor.readSysParam(&param);
        printf("read_sys_param: 0x%02x %s\n", ensure, sensor.ensureMessage(ensure));
        if (ensure == AS608::kOk) {
            printf("capacity=%u level=%u addr=0x%08x packet=%u baud_code=%u\n",
                   param.max_templates,
                   param.security_level,
                   param.address,
                   param.packet_size,
                   param.baud_rate);
        }

        uint16_t count = 0;
        ensure = sensor.validTemplateNum(&count);
        printf("template_count: 0x%02x %s count=%u\n",
               ensure, sensor.ensureMessage(ensure), count);
        return ensure == AS608::kOk ? 0 : 2;
    }

    if (strcmp(cmd, "search") == 0) {
        AS608::SearchResult result;
        uint8_t ensure = AS608::kTimeout;
        printf("Put finger on sensor...\n");
        if (sensor.identify(&result, 0, 300, &ensure)) {
            printf("match id=%u score=%u\n", result.page_id, result.match_score);
            return 0;
        }
        printf("search failed: 0x%02x %s\n", ensure, sensor.ensureMessage(ensure));
        return 2;
    }

    if (strcmp(cmd, "enroll") == 0 && argc > cmd_index + 1) {
        uint16_t id = (uint16_t)atoi(argv[cmd_index + 1]);
        uint8_t ensure = AS608::kTimeout;
        printf("Enroll id=%u. Press the same finger twice.\n", id);
        if (sensor.enroll(id, &ensure)) {
            printf("enroll ok id=%u\n", id);
            return 0;
        }
        printf("enroll failed: 0x%02x %s\n", ensure, sensor.ensureMessage(ensure));
        return 2;
    }

    if (strcmp(cmd, "delete") == 0 && argc > cmd_index + 1) {
        uint16_t id = (uint16_t)atoi(argv[cmd_index + 1]);
        uint8_t ensure = AS608::kTimeout;
        if (sensor.remove(id, 1, &ensure)) {
            printf("delete ok id=%u\n", id);
            return 0;
        }
        printf("delete failed: 0x%02x %s\n", ensure, sensor.ensureMessage(ensure));
        return 2;
    }

    if (strcmp(cmd, "empty") == 0) {
        uint8_t ensure = sensor.empty();
        printf("empty: 0x%02x %s\n", ensure, sensor.ensureMessage(ensure));
        return ensure == AS608::kOk ? 0 : 2;
    }

    usage(argv[0]);
    return 1;
}
