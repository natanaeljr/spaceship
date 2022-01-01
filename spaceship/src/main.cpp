#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <variant>
#include <algorithm>

#include <glbinding/gl33core/gl.h>
#include <glbinding/glbinding.h>
#include <glbinding-aux/debug.h>
using namespace gl;

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <GLFW/glfw3.h>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

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

template<typename T>
struct MoveNum {
  T num;
  MoveNum(T n) : num(n) {}
  MoveNum(MoveNum&& o) : num(o.num) { o.num = 0; }
  MoveNum& operator=(MoveNum&& o) { num = o.num; o.num = 0; return *this; }
  operator T() const { return num; }
};

/// Enumeration of supported Attributes
enum class GLAttr {
  POSITION,
  COLOR,
  MODEL,
  TEXCOORD,
  COUNT, // must be last
};

/// Enumeration of supported Uniforms
enum class GLUnif {
  COLOR,
  MODEL,
  VIEW,
  PROJECTION,
  TEXTURE0,
  COUNT, // must be last
};

/// GLShader represents an OpenGL shader program with utility functions
/// to build the shader, bind/unbind, get uniform and attribute location and trace requests/execution.
class GLShader final {
  /// Program name
  std::string name_;
  /// Program ID
  MoveNum<unsigned int> id_;
  /// Attributes' location
  std::array<GLint, static_cast<size_t>(GLAttr::COUNT)> attrs_ = { -1 };
  /// Uniforms' location
  std::array<GLint, static_cast<size_t>(GLAttr::COUNT)> unifs_ = { -1 };

  public:
  explicit GLShader(std::string name) : name_(std::move(name)), id_(glCreateProgram()) {
    TRACE("New GLShader program '{}' [{}]", name_, id_);
  }
  ~GLShader() {
    if (id_) {
      glDeleteProgram(id_);
      TRACE("Delete GLShader program '{}' [{}]", name_, id_);
    }
  }
  GLShader(GLShader&& o) = default;
  GLShader(const GLShader&) = delete;
  GLShader& operator=(GLShader&&) = default;
  GLShader& operator=(const GLShader&) = delete;

  public:
  /// Get shader program name
  [[nodiscard]] std::string_view name() const { return name_; }

  /// Bind shader program
  void bind() { glUseProgram(id_); }
  /// Unbind shader program
  void unbind() { glUseProgram(0); }

  /// Get attribute location
  [[nodiscard]] GLint attr_loc(GLAttr attr) const { return attrs_[static_cast<size_t>(attr)]; }
  /// Get uniform location
  [[nodiscard]] GLint unif_loc(GLUnif unif) const { return unifs_[static_cast<size_t>(unif)]; }

  /// Load attributes' location into local array
  void load_attr_loc(GLAttr attr, std::string_view attr_name)
  {
    const GLint new_loc = glGetAttribLocation(id_, attr_name.data());
    if (new_loc == -1) {
      CRITICAL("Failed to get location for attribute '{}' from GLShader '{}' [{}]", attr_name, name_, id_);
      std::abort();
    }
    attrs_[static_cast<size_t>(attr)] = new_loc;
  }

  /// Load uniforms' location into local array
  void load_unif_loc(GLUnif unif, std::string_view unif_name)
  {
    const GLint new_loc = glGetUniformLocation(id_, unif_name.data());
    if (new_loc == -1) {
      CRITICAL("Failed to get location for uniform '{}' from GLShader '{}' [{}]", unif_name, name_, id_);
      std::abort();
    }
    unifs_[static_cast<size_t>(unif)] = new_loc;
  }

  public:
  /// Build the shader program from sources
  static auto build(std::string name, std::string_view vert_src, std::string_view frag_src) -> std::optional<GLShader>
  {
    auto vertex = compile_shader(GL_VERTEX_SHADER, vert_src.data());
    auto fragment = compile_shader(GL_FRAGMENT_SHADER, frag_src.data());
    if (!vertex || !fragment) {
      ERROR("Failed to Compile Shaders");
      if (vertex) glDeleteShader(*vertex);
      if (fragment) glDeleteShader(*fragment);
      return std::nullopt;
    }
    auto shader = GLShader(name);
    if (!link_shader(shader.id_, *vertex, *fragment)) {
      ERROR("Failed to Link GLShader program");
      glDeleteShader(*vertex);
      glDeleteShader(*fragment);
      return std::nullopt;
    }
    glDeleteShader(*vertex);
    glDeleteShader(*fragment);
    DEBUG("Compiled & Linked shader program '{}' [{}]", shader.name_, shader.id_);
    return shader;
  }

  private:
  /// Compile single shader from sources
  static auto compile_shader(GLenum shader_type, const char* shader_src) -> std::optional<GLuint>
  {
    GLuint shader = glCreateShader(shader_type);
    glShaderSource(shader, 1, &shader_src, nullptr);
    glCompileShader(shader);
    GLint info_len = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &info_len);
    if (info_len) {
      auto info = std::make_unique<char[]>(info_len);
      glGetShaderInfoLog(shader, info_len, nullptr, info.get());
      DEBUG("{} Compilation Output:\n{}", shader_type_str(shader_type), info.get());
    }
    GLint compiled = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
      ERROR("Failed to Compile {}", shader_type_str(shader_type));
      glDeleteShader(shader);
      return std::nullopt;
    }
    return shader;
  }

  /// Link shaders into program object
  static bool link_shader(GLuint program, GLuint vert, GLuint frag)
  {
    glAttachShader(program, vert);
    glAttachShader(program, frag);
    glLinkProgram(program);
    GLint info_len = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &info_len);
    if (info_len) {
      auto info = std::make_unique<char[]>(info_len);
      glGetProgramInfoLog(program, info_len, nullptr, info.get());
      DEBUG("GLShader Program Link Output:\n{}", info.get());
    }
    GLint link_status = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &link_status);
    if (!link_status)
      ERROR("Failed to Link GLShader Program");
    glDetachShader(program, vert);
    glDetachShader(program, frag);
    return link_status;
  }

  /// Stringify opengl shader type.
  static auto shader_type_str(GLenum shader_type) -> std::string_view
  {
    switch (shader_type) {
      case GL_VERTEX_SHADER: return "GL_VERTEX_SHADER";
      case GL_FRAGMENT_SHADER: return "GL_FRAGMENT_SHADER";
      default: ASSERT_MSG(0, "Invalid shader type {}", shader_type); return "<invalid>";
    }
  }
};

auto load_color_shader() -> GLShader
{
  static constexpr std::string_view kColorShaderVert = R"(
#version 330 core
in vec3 aPosition;
in vec4 aColor;
out vec4 fColor;
uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;
void main()
{
  gl_Position = uProjection * uView * uModel * vec4(aPosition, 1.0);
  fColor = aColor;
}
)";

  static constexpr std::string_view kColorShaderFrag = R"(
#version 330 core
in vec4 fColor;
out vec4 outColor;
void main()
{
  outColor = fColor;
}
)";

  auto color_shader = GLShader::build("ColorShader", kColorShaderVert, kColorShaderFrag);
  ASSERT(color_shader);
  color_shader->bind();
  color_shader->load_attr_loc(GLAttr::POSITION, "aPosition");
  color_shader->load_attr_loc(GLAttr::COLOR, "aColor");
  color_shader->load_unif_loc(GLUnif::MODEL, "uModel");
  color_shader->load_unif_loc(GLUnif::VIEW, "uView");
  color_shader->load_unif_loc(GLUnif::PROJECTION, "uProjection");

  return std::move(*color_shader);
}

struct Shaders {
  GLShader color_shader;
};

Shaders load_shaders()
{
  return {
    .color_shader = load_color_shader(),
  };
}

struct GLObject {
  GLuint vbo;
  GLuint ebo;
  GLuint vao;
  size_t num_indices;
};

struct Vertex {
  glm::vec3 pos;
  glm::vec4 color;
};

// (-1,+1)       (+1,+1)
//  Y ^ - - - - - - o
//    |  B       D  |
//    |   +-----+   |
//    |   | \   |   |
//    |   |  0  |   |
//    |   |   \ |   |
//    |   +-----+   |
//    |  A       C  |
//    o - - - - - - > X
// (-1,-1)       (+1,-1)
// positive Z goes through screen towards you

static constexpr Vertex kQuadVertices[] = {
    { .pos = { -1.0f, -1.0f, +0.0f }, .color = { 1.0f, 0.0f, 0.0f, 1.0f } },
    { .pos = { -1.0f, +1.0f, +0.0f }, .color = { 1.0f, 0.0f, 1.0f, 1.0f } },
    { .pos = { +1.0f, -1.0f, +0.0f }, .color = { 0.0f, 1.0f, 0.0f, 1.0f } },
    { .pos = { +1.0f, +1.0f, +0.0f }, .color = { 0.0f, 0.0f, 1.0f, 1.0f } },
};

static constexpr GLushort kQuadIndices[] = {
    0, 1, 2,
    2, 1, 3,
};

GLObject create_colored_quad(const GLShader& shader)
{
  GLuint vbo, ebo, vao;
  glGenBuffers(1, &vbo);
  glGenBuffers(1, &ebo);
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(kQuadVertices), kQuadVertices, GL_STATIC_DRAW);
  glEnableVertexAttribArray(shader.attr_loc(GLAttr::POSITION));
  glVertexAttribPointer(shader.attr_loc(GLAttr::POSITION), 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*) offsetof(Vertex, pos));
  glEnableVertexAttribArray(shader.attr_loc(GLAttr::COLOR));
  glVertexAttribPointer(shader.attr_loc(GLAttr::COLOR), 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*) offsetof(Vertex, color));
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(kQuadIndices), kQuadIndices, GL_STATIC_DRAW);
  glBindVertexArray(0);
  return { vbo, ebo, vao, 6 };
}

struct Transform {
  glm::vec3 position;
  glm::vec3 scale;
  glm::quat rotation;
};

void render_object(const GLShader& shader, const GLObject& obj)
{
  auto transform = Transform{
    .position = glm::vec3(0.0f, 0.0f, 0.0f),
    .scale = glm::vec3(0.4f),
    .rotation = glm::quat(1.0f, glm::vec3(0.0f)),
  };

  float dt = glfwGetTime();
  transform.position.y = std::sin(dt) * 0.4f;

  glm::mat4 translation = glm::translate(glm::mat4(1.0f), transform.position);
  glm::mat4 rotation = glm::toMat4(transform.rotation);
  glm::mat4 scale = glm::scale(glm::mat4(1.0f), transform.scale);
  glm::mat4 model = translation * rotation * scale;

  glUniformMatrix4fv(shader.unif_loc(GLUnif::MODEL), 1, GL_FALSE, glm::value_ptr(model));
  glBindVertexArray(obj.vao);
  glDrawElements(GL_TRIANGLES, obj.num_indices, GL_UNSIGNED_SHORT, nullptr);
}

struct Scene {
  GLObject quad;
};

struct Engine {
  std::optional<Shaders> shaders;
  std::optional<Scene> scene;
};

int game_init(Engine& engine)
{
  engine.shaders = load_shaders();
  engine.scene = Scene{};
  engine.scene->quad = create_colored_quad(engine.shaders->color_shader);
  DEBUG("Generated colored Quad");

  return 0;
}

int game_update(Engine& engine)
{
  return 0;
}

int game_render(Engine& engine)
{
  glEnable(GL_DEPTH_TEST);
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  glClearColor(0.2f, 0.1f, 0.1f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  GLShader& color_shader = engine.shaders->color_shader;
  color_shader.bind();

  auto view = glm::mat4(1.0f);
  auto projection = glm::mat4(1.0f);
  glUniformMatrix4fv(color_shader.unif_loc(GLUnif::VIEW), 1, GL_FALSE, glm::value_ptr(view));
  glUniformMatrix4fv(color_shader.unif_loc(GLUnif::PROJECTION), 1, GL_FALSE, glm::value_ptr(projection));

  render_object(color_shader, engine.scene->quad);

  return 0;
}

int game_loop(GLFWwindow* window)
{
  Engine engine;
  game_init(engine);
  while (!glfwWindowShouldClose(window)) {
    game_update(engine);
    game_render(engine);
    glfwSwapBuffers(window);
    glfwPollEvents();
  }
  return 0;
}

int create_window(GLFWwindow*& window)
{
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  window = glfwCreateWindow(480, 720, "Spaceship", nullptr, nullptr);
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
  //glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
  glfwSwapInterval(1);  // enable/disable vsync

  return 0;
}

int main(int argc, char *argv[])
{
  int ret = 0;
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
  INFO("Initializing..");

  // Create Window =============================================================
  INFO("Creating Window..");
  GLFWwindow* window = nullptr;
  ret = create_window(window);
  if (ret) return ret;

  // Load GL ===================================================================
  INFO("Loading OpenGL..");
  glbinding::initialize(glfwGetProcAddress);
  glbinding::aux::enableGetErrorCallback();

  // Game Loop =================================================================
  INFO("Game Loop..");
  ret = game_loop(window);

  // End =======================================================================
  INFO("Terminating..");
  glfwTerminate();

  INFO("Exit");
  return ret;
}
