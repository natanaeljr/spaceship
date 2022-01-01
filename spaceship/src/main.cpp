#include <cstdio>
#include <cstdlib>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#define CRITICAL SPDLOG_CRITICAL
#define ERROR SPDLOG_ERROR
#define WARN SPDLOG_WARN
#define INFO SPDLOG_INFO
#define DEBUG SPDLOG_DEBUG
#define TRACE SPDLOG_TRACE

#define ASSERT_MSG(cond, ...)                                                  \
  do {                                                                         \
    if (cond) {                                                                \
    } else {                                                                   \
      CRITICAL(__VA_ARGS__);                                                   \
      abort();                                                                 \
    }                                                                          \
  } while (0)
#define ASSERT(cond) ASSERT_MSG(cond, "Assertion failed: ({})", #cond)

int main(int argc, char *argv[])
{
  auto log_level = spdlog::level::trace;

  // Parse arguments
  for (int argi = 1; argi < argc; ++argi) {
    if (!strcmp(argv[argi], "--log")) {
      argi++;
      if (argi < argc) {
        log_level = spdlog::level::from_str(argv[argi]);
      } else {
        fprintf(stderr, "--log: missing argument\n");
        return -2;
      }
    } else {
      fprintf(stderr, "unknown argument: %s\n", argv[argi]);
      return -1;
    }
  }

  spdlog::set_default_logger(spdlog::stdout_color_mt("main"));
  spdlog::set_pattern("%Y-%m-%d %T.%e <%^%l%$> [%n] %s:%#: %!() -> %v");
  spdlog::set_level(log_level);

  INFO("Initializing...");

  INFO("Terminating");

  return 0;
}
