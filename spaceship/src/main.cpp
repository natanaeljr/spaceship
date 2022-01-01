#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <glbinding/gl33core/gl.h>
#include <glbinding/glbinding.h>
using namespace gl;

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <GLFW/glfw3.h>

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

  // Parse Arguments ===========================================================
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

  // Setup Logging =============================================================
  spdlog::set_default_logger(spdlog::stdout_color_mt("main"));
  spdlog::set_pattern("%Y-%m-%d %T.%e <%^%l%$> [%n] %s:%#: %!() -> %v");
  spdlog::set_level(log_level);

  INFO("Initializing...");

  // Spawn Window ==============================================================
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWwindow* window = glfwCreateWindow(480, 720, "Spaceship", nullptr, nullptr);
  if (window == nullptr) {
    std::cerr << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return -3;
  }
  glfwMakeContextCurrent(window);
  //glfwSetWindowUserPointer(window, &state);
  // callbacks
  //glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
  //glfwSetKeyCallback(window, key_callback);
  //glfwSetCursorPosCallback(window, cursor_position_callback);
  //glfwSetMouseButtonCallback(window, mouse_button_callback);
  //glfwSetScrollCallback(window, scroll_callback);
  // settings
  int width, height;
  glfwGetWindowSize(window, &width, &height);
  glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
  glfwSwapInterval(0);  // disable vsync

  glbinding::initialize(glfwGetProcAddress);

  while (!glfwWindowShouldClose(window)) {
    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glfwTerminate();

  INFO("Terminating");

  return 0;
}
