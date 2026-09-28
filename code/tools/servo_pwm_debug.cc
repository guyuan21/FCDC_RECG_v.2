// Standalone sysfs PWM servo debug tool for Luckfox Pico Pro Max.
//
// Default mapping:
//   servo2: /sys/class/pwm/pwmchip2/pwm0, 180-degree positional servo
//   servo1: /sys/class/pwm/pwmchip6/pwm0, 180-degree positional servo
//
// Typical hobby servos use a 20 ms period and a pulse width around
// 500-2500 us. Adjust --min-us/--max-us/--neutral-us for your hardware.

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

static volatile sig_atomic_t g_stop = 0;

struct Options {
    std::string servo1_chip = "/sys/class/pwm/pwmchip6";
    std::string servo2_chip = "/sys/class/pwm/pwmchip2";
    int servo1_channel = 0;
    int servo2_channel = 0;
    int period_us = 20000;
    int min_us = 500;
    int max_us = 2500;
    int neutral_us = 1500;
};

struct PwmDevice {
    std::string chip_path;
    int channel;
};

static void on_signal(int)
{
    g_stop = 1;
}

static bool path_exists(const std::string &path)
{
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

static bool write_text(const std::string &path, const std::string &text)
{
    int fd = open(path.c_str(), O_WRONLY);
    if (fd < 0) {
        fprintf(stderr, "open %s failed: %s\n", path.c_str(), strerror(errno));
        return false;
    }

    const char *data = text.c_str();
    size_t left = text.size();
    while (left > 0) {
        ssize_t n = write(fd, data, left);
        if (n < 0) {
            fprintf(stderr, "write %s failed: %s\n", path.c_str(), strerror(errno));
            close(fd);
            return false;
        }
        data += n;
        left -= (size_t)n;
    }

    close(fd);
    return true;
}

static bool write_int64(const std::string &path, long long value)
{
    char buf[64];
    snprintf(buf, sizeof(buf), "%lld", value);
    return write_text(path, buf);
}

static std::string pwm_path(const PwmDevice &pwm)
{
    char buf[32];
    snprintf(buf, sizeof(buf), "/pwm%d", pwm.channel);
    return pwm.chip_path + buf;
}

static bool export_pwm(const PwmDevice &pwm)
{
    if (!path_exists(pwm.chip_path)) {
        fprintf(stderr, "PWM chip not found: %s\n", pwm.chip_path.c_str());
        return false;
    }

    const std::string path = pwm_path(pwm);
    if (path_exists(path)) {
        return true;
    }

    if (!write_int64(pwm.chip_path + "/export", pwm.channel)) {
        return false;
    }

    for (int i = 0; i < 20; ++i) {
        if (path_exists(path)) {
            return true;
        }
        usleep(10000);
    }

    fprintf(stderr, "PWM path did not appear after export: %s\n", path.c_str());
    return false;
}

static bool set_enable(const PwmDevice &pwm, bool enable)
{
    return write_int64(pwm_path(pwm) + "/enable", enable ? 1 : 0);
}

static bool configure_pulse_us(const PwmDevice &pwm, int period_us, int pulse_us)
{
    if (!export_pwm(pwm)) {
        return false;
    }

    const std::string path = pwm_path(pwm);
    const long long period_ns = (long long)period_us * 1000LL;
    const long long duty_ns = (long long)pulse_us * 1000LL;

    // Many sysfs PWM drivers require period changes while disabled.
    set_enable(pwm, false);

    // Hobby servos expect a normal active-high pulse.
    write_text(path + "/polarity", "normal");

    if (!write_int64(path + "/period", period_ns)) {
        return false;
    }
    if (!write_int64(path + "/duty_cycle", duty_ns)) {
        return false;
    }
    if (!set_enable(pwm, true)) {
        return false;
    }

    printf("%s: period=%dus pulse=%dus enabled\n",
           path.c_str(), period_us, pulse_us);
    return true;
}

static bool disable_pwm(const PwmDevice &pwm)
{
    if (!export_pwm(pwm)) {
        return false;
    }
    if (!set_enable(pwm, false)) {
        return false;
    }
    printf("%s: disabled\n", pwm_path(pwm).c_str());
    return true;
}

static int clamp_int(int value, int lo, int hi)
{
    return std::max(lo, std::min(value, hi));
}

static int angle_to_pulse_us(int angle, int max_angle, const Options &opt)
{
    angle = clamp_int(angle, 0, max_angle);
    return opt.min_us + (opt.max_us - opt.min_us) * angle / max_angle;
}

static int speed_to_pulse_us(int speed, const Options &opt)
{
    speed = clamp_int(speed, -100, 100);
    if (speed >= 0) {
        return opt.neutral_us + (opt.max_us - opt.neutral_us) * speed / 100;
    }
    return opt.neutral_us + (opt.neutral_us - opt.min_us) * speed / 100;
}

static bool parse_int(const char *text, int *value)
{
    if (!text || !text[0]) {
        return false;
    }

    char *end = nullptr;
    long parsed = strtol(text, &end, 10);
    if (!end || *end != '\0') {
        return false;
    }
    if (parsed < -2147483647L || parsed > 2147483647L) {
        return false;
    }

    *value = (int)parsed;
    return true;
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage:\n"
            "  %s interactive\n"
            "  %s servo2 <angle 0-180>\n"
            "  %s servo1 <angle 0-180>\n"
            "  %s servo1-speed <speed -100..100> [seconds]\n"
            "  %s pulse <servo1|servo2> <pulse_us>\n"
            "  %s center\n"
            "  %s off\n"
            "\n"
            "Options before command:\n"
            "  --servo1-chip <path>      default /sys/class/pwm/pwmchip6\n"
            "  --servo2-chip <path>      default /sys/class/pwm/pwmchip2\n"
            "  --servo1-channel <n>      default 0\n"
            "  --servo2-channel <n>      default 0\n"
            "  --period-us <n>           default 20000\n"
            "  --min-us <n>              default 500\n"
            "  --max-us <n>              default 2500\n"
            "  --neutral-us <n>          default 1500\n"
            "\n"
            "Interactive commands:\n"
            "  s1 <angle 0-180>          set servo1 angle\n"
            "  s1speed <speed -100..100> set servo1 as continuous-rotation servo\n"
            "  s2 <angle 0-180>          set servo2 angle\n"
            "  p1 <pulse_us>             set servo1 pulse directly\n"
            "  p2 <pulse_us>             set servo2 pulse directly\n"
            "  center                    set both servos to neutral pulse\n"
            "  off                       disable both PWM outputs\n"
            "  quit                      exit\n",
            prog, prog, prog, prog, prog, prog, prog);
}

static std::vector<std::string> split_words(const std::string &line)
{
    std::vector<std::string> out;
    size_t i = 0;
    while (i < line.size()) {
        while (i < line.size() && isspace((unsigned char)line[i])) {
            ++i;
        }
        size_t start = i;
        while (i < line.size() && !isspace((unsigned char)line[i])) {
            ++i;
        }
        if (start < i) {
            out.push_back(line.substr(start, i - start));
        }
    }
    return out;
}

static bool handle_interactive_command(const std::vector<std::string> &args,
                                       const Options &opt,
                                       const PwmDevice &servo1,
                                       const PwmDevice &servo2)
{
    if (args.empty()) {
        return true;
    }

    int value = 0;
    const std::string &cmd = args[0];
    if ((cmd == "quit") || (cmd == "q") || (cmd == "exit")) {
        return false;
    }
    if (cmd == "help" || cmd == "?") {
        usage("servo_pwm_debug");
        return true;
    }
    if (cmd == "s1" && args.size() == 2 && parse_int(args[1].c_str(), &value)) {
        configure_pulse_us(servo1, opt.period_us, angle_to_pulse_us(value, 180, opt));
        return true;
    }
    if (cmd == "s1speed" && args.size() == 2 && parse_int(args[1].c_str(), &value)) {
        configure_pulse_us(servo1, opt.period_us, speed_to_pulse_us(value, opt));
        return true;
    }
    if (cmd == "s2" && args.size() == 2 && parse_int(args[1].c_str(), &value)) {
        configure_pulse_us(servo2, opt.period_us, angle_to_pulse_us(value, 180, opt));
        return true;
    }
    if (cmd == "p1" && args.size() == 2 && parse_int(args[1].c_str(), &value)) {
        configure_pulse_us(servo1, opt.period_us, clamp_int(value, 0, opt.period_us));
        return true;
    }
    if (cmd == "p2" && args.size() == 2 && parse_int(args[1].c_str(), &value)) {
        configure_pulse_us(servo2, opt.period_us, clamp_int(value, 0, opt.period_us));
        return true;
    }
    if (cmd == "center") {
        configure_pulse_us(servo1, opt.period_us, opt.neutral_us);
        configure_pulse_us(servo2, opt.period_us, opt.neutral_us);
        return true;
    }
    if (cmd == "off") {
        disable_pwm(servo1);
        disable_pwm(servo2);
        return true;
    }

    fprintf(stderr, "Unknown command. Type help for usage.\n");
    return true;
}

static int interactive_loop(const Options &opt,
                            const PwmDevice &servo1,
                            const PwmDevice &servo2)
{
    printf("Servo PWM interactive debug\n");
    printf("servo1: %s/pwm%d, servo2: %s/pwm%d\n",
           opt.servo1_chip.c_str(), opt.servo1_channel,
           opt.servo2_chip.c_str(), opt.servo2_channel);
    printf("Type help for commands.\n");

    char line[256];
    while (!g_stop) {
        printf("> ");
        fflush(stdout);

        if (!fgets(line, sizeof(line), stdin)) {
            break;
        }

        std::vector<std::string> args = split_words(line);
        if (!handle_interactive_command(args, opt, servo1, servo2)) {
            break;
        }
    }

    return 0;
}

static bool parse_options(int argc, char **argv, Options *opt, int *cmd_index)
{
    int i = 1;
    while (i < argc) {
        std::string arg = argv[i];
        if (arg.size() < 2 || arg[0] != '-' || arg[1] != '-') {
            break;
        }
        if (arg == "--help") {
            *cmd_index = i;
            return true;
        }
        if (i + 1 >= argc) {
            fprintf(stderr, "Missing value for %s\n", argv[i]);
            return false;
        }

        if (arg == "--servo1-chip") {
            opt->servo1_chip = argv[++i];
        } else if (arg == "--servo2-chip") {
            opt->servo2_chip = argv[++i];
        } else if (arg == "--servo1-channel") {
            if (!parse_int(argv[++i], &opt->servo1_channel)) return false;
        } else if (arg == "--servo2-channel") {
            if (!parse_int(argv[++i], &opt->servo2_channel)) return false;
        } else if (arg == "--period-us") {
            if (!parse_int(argv[++i], &opt->period_us)) return false;
        } else if (arg == "--min-us") {
            if (!parse_int(argv[++i], &opt->min_us)) return false;
        } else if (arg == "--max-us") {
            if (!parse_int(argv[++i], &opt->max_us)) return false;
        } else if (arg == "--neutral-us") {
            if (!parse_int(argv[++i], &opt->neutral_us)) return false;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return false;
        }
        ++i;
    }

    if (opt->period_us <= 0 || opt->min_us < 0 || opt->max_us <= opt->min_us ||
        opt->neutral_us < opt->min_us || opt->neutral_us > opt->max_us ||
        opt->max_us >= opt->period_us ||
        opt->servo1_channel < 0 || opt->servo2_channel < 0) {
        fprintf(stderr, "Invalid PWM timing or channel options\n");
        return false;
    }

    *cmd_index = i;
    return true;
}

int main(int argc, char **argv)
{
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    Options opt;
    int cmd_index = 1;
    if (!parse_options(argc, argv, &opt, &cmd_index) || cmd_index >= argc) {
        usage(argv[0]);
        return 1;
    }
    if (strcmp(argv[cmd_index], "--help") == 0) {
        usage(argv[0]);
        return 0;
    }

    PwmDevice servo1{opt.servo1_chip, opt.servo1_channel};
    PwmDevice servo2{opt.servo2_chip, opt.servo2_channel};

    const std::string cmd = argv[cmd_index];
    int value = 0;

    if (cmd == "interactive") {
        return interactive_loop(opt, servo1, servo2);
    }
    if (cmd == "servo2" && argc > cmd_index + 1 &&
        parse_int(argv[cmd_index + 1], &value)) {
        return configure_pulse_us(servo2, opt.period_us,
                                  angle_to_pulse_us(value, 180, opt)) ? 0 : 2;
    }
    if (cmd == "servo1" && argc > cmd_index + 1 &&
        parse_int(argv[cmd_index + 1], &value)) {
        return configure_pulse_us(servo1, opt.period_us,
                                  angle_to_pulse_us(value, 180, opt)) ? 0 : 2;
    }
    if (cmd == "servo1-speed" && argc > cmd_index + 1 &&
        parse_int(argv[cmd_index + 1], &value)) {
        if (!configure_pulse_us(servo1, opt.period_us, speed_to_pulse_us(value, opt))) {
            return 2;
        }
        if (argc > cmd_index + 2) {
            int seconds = 0;
            if (!parse_int(argv[cmd_index + 2], &seconds) || seconds < 0) {
                usage(argv[0]);
                return 1;
            }
            sleep((unsigned int)seconds);
            return configure_pulse_us(servo1, opt.period_us, opt.neutral_us) ? 0 : 2;
        }
        return 0;
    }
    if (cmd == "pulse" && argc > cmd_index + 2 &&
        parse_int(argv[cmd_index + 2], &value)) {
        const std::string which = argv[cmd_index + 1];
        if (which == "servo1") {
            return configure_pulse_us(servo1, opt.period_us,
                                      clamp_int(value, 0, opt.period_us)) ? 0 : 2;
        }
        if (which == "servo2") {
            return configure_pulse_us(servo2, opt.period_us,
                                      clamp_int(value, 0, opt.period_us)) ? 0 : 2;
        }
    }
    if (cmd == "center") {
        bool ok1 = configure_pulse_us(servo1, opt.period_us, opt.neutral_us);
        bool ok2 = configure_pulse_us(servo2, opt.period_us, opt.neutral_us);
        return (ok1 && ok2) ? 0 : 2;
    }
    if (cmd == "off") {
        bool ok1 = disable_pwm(servo1);
        bool ok2 = disable_pwm(servo2);
        return (ok1 && ok2) ? 0 : 2;
    }

    usage(argv[0]);
    return 1;
}
