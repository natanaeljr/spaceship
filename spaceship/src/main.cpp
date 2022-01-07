#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <variant>
#include <algorithm>
#include <unordered_map>
#include <functional>
#include <fstream>

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
#include <stb/stb_image.h>
#include <stb/stb_rect_pack.h>
#include <stb/stb_truetype.h>
#include <gsl/span>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Utils

#define TRACE SPDLOG_TRACE
#define DEBUG SPDLOG_DEBUG
#define INFO SPDLOG_INFO
#define WARN SPDLOG_WARN
#define ERROR SPDLOG_ERROR
#define CRITICAL SPDLOG_CRITICAL
#define ABORT_MSG(...)         \
  do {                         \
    CRITICAL(__VA_ARGS__);     \
    std::abort();              \
  } while (0)
#define ASSERT_MSG(cond, ...)  \
  do {                         \
    if (cond) {                \
    } else {                   \
      ABORT_MSG(__VA_ARGS__);  \
    }                          \
  } while (0)
#define ASSERT(cond) ASSERT_MSG(cond, "Assertion failed: ({})", #cond)
#define ASSERT_OPT(opt) [&]{ auto ret = (opt); ASSERT(ret); return std::move(*ret); }()

using namespace std::string_literals;

/// Read file contents to a string
auto read_file_to_string(const std::string& filename) -> std::optional<std::string>
{
  std::string string;
  std::fstream fstream(filename);
  if (!fstream) {
    ERROR("Failed to open file ({}): {}", filename, std::strerror(errno));
    return std::nullopt;
  }
  fstream.seekg(0, std::ios::end);
  string.reserve(fstream.tellg());
  fstream.seekg(0, std::ios::beg);
  string.assign((std::istreambuf_iterator<char>(fstream)), std::istreambuf_iterator<char>());
  return string;
}

/// Utility type to make unique numbers (IDs) movable, when moved the value should be zero
template<typename T>
struct UniqueNum {
  T inner;
  UniqueNum(T n) : inner(n) {}
  UniqueNum(UniqueNum&& o) : inner(o.inner) { o.inner = 0; }
  UniqueNum& operator=(UniqueNum&& o) { inner = o.inner; o.inner = 0; return *this; }
  UniqueNum(const UniqueNum&) = delete;
  UniqueNum& operator=(const UniqueNum&) = delete;
  operator T() const { return inner; }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Shaders

/// Enumeration of supported Shader Attributes
enum class GLAttr {
  POSITION,
  COLOR,
  MODEL,
  TEXCOORD,
  COUNT, // must be last
};

/// Enumeration of supported Shader Uniforms
enum class GLUnif {
  COLOR,
  OUTLINE_COLOR,
  OUTLINE_THICKNESS,
  MODEL,
  VIEW,
  PROJECTION,
  TEXTURE0,
  SUBROUTINE,
  COUNT, // must be last
};

/// Enumeration of supported Shader Subroutines
enum class GLSub {
  TEXTURE,
  FONT,
  COLOR,
};

/// GLShader represents an OpenGL shader program
class GLShader final {
  /// Program name
  std::string name_;
  /// Program ID
  UniqueNum<unsigned int> id_;
  /// Attributes' location
  std::array<GLint, static_cast<size_t>(GLAttr::COUNT)> attrs_ = { -1 };
  /// Uniforms' location
  std::array<GLint, static_cast<size_t>(GLUnif::COUNT)> unifs_ = { -1 };

  public:
  explicit GLShader(std::string name) : name_(std::move(name)), id_(glCreateProgram()) {
    TRACE("New GLShader program '{}'[{}]", name_, id_);
  }
  ~GLShader() {
    if (id_) {
      glDeleteProgram(id_);
      TRACE("Delete GLShader program '{}'[{}]", name_, id_);
    }
  }
  // Movable but not Copyable
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
    const GLint loc = glGetAttribLocation(id_, attr_name.data());
    if (loc == -1)
      ABORT_MSG("Failed to get location for attribute '{}' GLShader '{}'[{}]", attr_name, name_, id_);
    TRACE("Loaded attritube '{}' location {} GLShader '{}'[{}]", attr_name, loc, name_, id_);
    attrs_[static_cast<size_t>(attr)] = loc;
  }

  /// Load uniforms' location into local array
  void load_unif_loc(GLUnif unif, std::string_view unif_name)
  {
    const GLint loc = glGetUniformLocation(id_, unif_name.data());
    if (loc == -1)
      ABORT_MSG("Failed to get location for uniform '{}' GLShader '{}'[{}]", unif_name, name_, id_);
    TRACE("Loaded uniform '{}' location {} GLShader '{}'[{}]", unif_name, loc, name_, id_);
    unifs_[static_cast<size_t>(unif)] = loc;
  }

  public:
  /// Build a shader program from sources
  static auto build(std::string name, std::string_view vert_src, std::string_view frag_src) -> std::optional<GLShader>
  {
    auto shader = GLShader(std::move(name));
    auto vertex = shader.compile(GL_VERTEX_SHADER, vert_src.data());
    auto fragment = shader.compile(GL_FRAGMENT_SHADER, frag_src.data());
    if (!vertex || !fragment) {
      ERROR("Failed to Compile Shaders for program '{}'[{}]", shader.name_, shader.id_);
      if (vertex) glDeleteShader(*vertex);
      if (fragment) glDeleteShader(*fragment);
      return std::nullopt;
    }
    if (!shader.link(*vertex, *fragment)) {
      ERROR("Failed to Link GLShader program '{}'[{}]", shader.name_, shader.id_);
      glDeleteShader(*vertex);
      glDeleteShader(*fragment);
      return std::nullopt;
    }
    glDeleteShader(*vertex);
    glDeleteShader(*fragment);
    TRACE("Compiled&Linked shader program '{}'[{}]", shader.name_, shader.id_);
    return shader;
  }

  private:
  /// Compile a single shader from sources
  auto compile(GLenum shader_type, const char* shader_src) -> std::optional<GLuint>
  {
    GLuint shader = glCreateShader(shader_type);
    glShaderSource(shader, 1, &shader_src, nullptr);
    glCompileShader(shader);
    GLint info_len = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &info_len);
    if (info_len) {
      auto info = std::make_unique<char[]>(info_len);
      glGetShaderInfoLog(shader, info_len, nullptr, info.get());
      DEBUG("GLShader '{}'[{}] Compilation Output {}:\n{}", name_, id_, shader_type_str(shader_type), info.get());
    }
    GLint compiled = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
      ERROR("Failed to Compile {} for GLShader '{}'[{}]", shader_type_str(shader_type), name_, id_);
      glDeleteShader(shader);
      return std::nullopt;
    }
    return shader;
  }

  /// Link shaders into program object
  bool link(GLuint vert, GLuint frag)
  {
    glAttachShader(id_, vert);
    glAttachShader(id_, frag);
    glLinkProgram(id_);
    GLint info_len = 0;
    glGetProgramiv(id_, GL_INFO_LOG_LENGTH, &info_len);
    if (info_len) {
      auto info = std::make_unique<char[]>(info_len);
      glGetProgramInfoLog(id_, info_len, nullptr, info.get());
      DEBUG("GLShader '{}'[{}] Program Link Output:\n{}", name_, id_, info.get());
    }
    GLint link_status = 0;
    glGetProgramiv(id_, GL_LINK_STATUS, &link_status);
    if (!link_status)
      ERROR("Failed to Link GLShader Program '{}'[{}]", name_, id_);
    glDetachShader(id_, vert);
    glDetachShader(id_, frag);
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

/// Load Generic Shader
/// (supports rendering: Colored objects, Textured objects and BitmapFont text)
auto load_generic_shader() -> GLShader
{
  static constexpr std::string_view kShaderVert = R"(
#version 330 core
in vec3 aPosition;
in vec2 aTexCoord;
in vec4 aColor;
out vec4 fColor;
out vec2 fTexCoord;
uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;
void main()
{
  gl_Position = uProjection * uView * uModel * vec4(aPosition, 1.0);
  fTexCoord = aTexCoord;
  fColor = aColor;
}
)";

  static constexpr std::string_view kShaderFrag = R"(
#version 330 core
in vec2 fTexCoord;
in vec4 fColor;
out vec4 outColor;
uniform sampler2D uTexture0;
uniform vec4 uColor;
uniform vec4 uOutlineColor;
uniform float uOutlineThickness;
uniform int uSubRoutine;
vec4 font_color();
void main()
{
  if (uSubRoutine == 0) {
    outColor = texture(uTexture0, fTexCoord);
  } else if (uSubRoutine == 1) {
    outColor = font_color();
  } else if (uSubRoutine == 2) {
    outColor = fColor;
  } else {
    outColor = uColor;
  }
}
vec4 font_color()
{
  vec2 Offset = 1.0 / textureSize(uTexture0, 0) * uOutlineThickness;
  vec4 n = texture2D(uTexture0, vec2(fTexCoord.x, fTexCoord.y - Offset.y));
  vec4 e = texture2D(uTexture0, vec2(fTexCoord.x + Offset.x, fTexCoord.y));
  vec4 s = texture2D(uTexture0, vec2(fTexCoord.x, fTexCoord.y + Offset.y));
  vec4 w = texture2D(uTexture0, vec2(fTexCoord.x - Offset.x, fTexCoord.y));
  vec4 TexColor = vec4(vec3(1.0), texture2D(uTexture0, fTexCoord).r);
  float GrowedAlpha = TexColor.a;
  GrowedAlpha = mix(GrowedAlpha, 1.0, s.r);
  GrowedAlpha = mix(GrowedAlpha, 1.0, w.r);
  GrowedAlpha = mix(GrowedAlpha, 1.0, n.r);
  GrowedAlpha = mix(GrowedAlpha, 1.0, e.r);
  vec4 OutlineColorWithNewAlpha = vec4(uOutlineColor.rgb, uOutlineColor.a * GrowedAlpha);
  vec4 CharColor = TexColor * uColor;
  return mix(OutlineColorWithNewAlpha, CharColor, CharColor.a);
}
)";

  DEBUG("Loading Generic Shader");
  auto shader = GLShader::build("GenericShader", kShaderVert, kShaderFrag);
  ASSERT(shader);
  shader->bind();
  shader->load_attr_loc(GLAttr::POSITION, "aPosition");
  shader->load_attr_loc(GLAttr::TEXCOORD, "aTexCoord");
  shader->load_attr_loc(GLAttr::COLOR, "aColor");
  shader->load_unif_loc(GLUnif::MODEL, "uModel");
  shader->load_unif_loc(GLUnif::VIEW, "uView");
  shader->load_unif_loc(GLUnif::PROJECTION, "uProjection");
  shader->load_unif_loc(GLUnif::TEXTURE0, "uTexture0");
  shader->load_unif_loc(GLUnif::COLOR, "uColor");
  shader->load_unif_loc(GLUnif::OUTLINE_COLOR, "uOutlineColor");
  shader->load_unif_loc(GLUnif::OUTLINE_THICKNESS, "uOutlineThickness");
  shader->load_unif_loc(GLUnif::SUBROUTINE, "uSubRoutine");

  return std::move(*shader);
}

/// Holds the shaders used by the game
struct Shaders {
  GLShader generic_shader;
};

/// Loads all shaders used by the game
Shaders load_shaders()
{
  return {
    .generic_shader = load_generic_shader(),
  };
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// GLObjects

/// Vertex representation for a Colored Object
struct ColorVertex {
  glm::vec3 pos;
  glm::vec4 color;
};

/// Vertex representation for a Textured Object
struct TextureVertex {
  glm::vec3 pos;
  glm::vec2 texcoord;
};

/// Represents an object loaded to GPU memory that's renderable using indices
struct GLObject {
  UniqueNum<GLuint> vbo;
  UniqueNum<GLuint> ebo;
  UniqueNum<GLuint> vao;
  size_t num_indices;
  size_t num_vertices;

  ~GLObject() {
    if (vbo) glDeleteBuffers(1, &vbo.inner);
    if (ebo) glDeleteBuffers(1, &ebo.inner);
    if (vao) glDeleteVertexArrays(1, &vao.inner);
  }

  // Movable but not Copyable
  GLObject(GLObject&&) = default;
  GLObject(const GLObject&) = delete;
  GLObject& operator=(GLObject&&) = default;
  GLObject& operator=(const GLObject&) = delete;
};

/// Upload new Colored Indexed-Vertex object to GPU memory
GLObject create_colored_globject(const GLShader& shader, gsl::span<const ColorVertex> vertices, gsl::span<const GLushort> indices, GLenum usage = GL_STATIC_DRAW)
{
  GLuint vbo, ebo, vao;
  glGenBuffers(1, &vbo);
  glGenBuffers(1, &ebo);
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, vertices.size_bytes(), vertices.data(), usage);
  glEnableVertexAttribArray(shader.attr_loc(GLAttr::POSITION));
  glVertexAttribPointer(shader.attr_loc(GLAttr::POSITION), 3, GL_FLOAT, GL_FALSE, sizeof(ColorVertex), (void*) offsetof(ColorVertex, pos));
  glEnableVertexAttribArray(shader.attr_loc(GLAttr::COLOR));
  glVertexAttribPointer(shader.attr_loc(GLAttr::COLOR), 4, GL_FLOAT, GL_FALSE, sizeof(ColorVertex), (void*) offsetof(ColorVertex, color));
  glDisableVertexAttribArray(shader.attr_loc(GLAttr::TEXCOORD));
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size_bytes(), indices.data(), usage);
  glBindVertexArray(0);
  return { vbo, ebo, vao, indices.size(), vertices.size() };
}

/// Upload new Textured Indexed-Vertex object to GPU memory
GLObject create_textured_globject(const GLShader& shader, gsl::span<const TextureVertex> vertices, gsl::span<const GLushort> indices, GLenum usage = GL_STATIC_DRAW)
{
  GLuint vbo, ebo, vao;
  glGenBuffers(1, &vbo);
  glGenBuffers(1, &ebo);
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, vertices.size_bytes(), vertices.data(), usage);
  glEnableVertexAttribArray(shader.attr_loc(GLAttr::POSITION));
  glVertexAttribPointer(shader.attr_loc(GLAttr::POSITION), 3, GL_FLOAT, GL_FALSE, sizeof(TextureVertex), (void*) offsetof(TextureVertex, pos));
  glEnableVertexAttribArray(shader.attr_loc(GLAttr::TEXCOORD));
  glVertexAttribPointer(shader.attr_loc(GLAttr::TEXCOORD), 2, GL_FLOAT, GL_FALSE, sizeof(TextureVertex), (void*) offsetof(TextureVertex, texcoord));
  glDisableVertexAttribArray(shader.attr_loc(GLAttr::COLOR));
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size_bytes(), indices.data(), usage);
  glBindVertexArray(0);
  return { vbo, ebo, vao, indices.size(), vertices.size() };
}

// Quad Vertices:
// (-1,+1)       (+1,+1)
//  Y ^ - - - - - - o
//    |  D       A  |
//    |   +-----+   |
//    |   | \   |   |
//    |   |  0  |   |
//    |   |   \ |   |
//    |   +-----+   |
//    |  C       B  |
//    o - - - - - - > X
// (-1,-1)       (+1,-1)
// positive Z goes through screen towards you

static constexpr ColorVertex kColorQuadVertices[] = {
  { .pos = { +1.0f, +1.0f, +0.0f }, .color = { 0.0f, 0.0f, 1.0f, 1.0f } },
  { .pos = { +1.0f, -1.0f, +0.0f }, .color = { 0.0f, 1.0f, 0.0f, 1.0f } },
  { .pos = { -1.0f, -1.0f, +0.0f }, .color = { 1.0f, 0.0f, 0.0f, 1.0f } },
  { .pos = { -1.0f, +1.0f, +0.0f }, .color = { 1.0f, 0.0f, 1.0f, 1.0f } },
};

static constexpr TextureVertex kTextureQuadVertices[] = {
  { .pos = { +1.0f, +1.0f, +0.0f }, .texcoord = { 1.0f, 1.0f } },
  { .pos = { +1.0f, -1.0f, +0.0f }, .texcoord = { 1.0f, 0.0f } },
  { .pos = { -1.0f, -1.0f, +0.0f }, .texcoord = { 0.0f, 0.0f } },
  { .pos = { -1.0f, +1.0f, +0.0f }, .texcoord = { 0.0f, 1.0f } },
};

static constexpr GLushort kQuadIndices[] = {
  0, 1, 3,
  1, 2, 3,
};

/// Upload new colored Quad object to GPU memory
GLObject create_colored_quad_globject(const GLShader& shader)
{
  return create_colored_globject(shader, kColorQuadVertices, kQuadIndices);
}

/// Upload new textured Quad object to GPU memory
GLObject create_textured_quad_globject(const GLShader& shader)
{
  return create_textured_globject(shader, kTextureQuadVertices, kQuadIndices);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Sprites

/// Generate quad vertices for a spritesheet texture with frames laid out linearly.
/// count=3:        .texcoord (U,V)
/// (0,1) +-----+-----+-----+ (1,1)
///       |     |     |     |
///       |  1  |  2  |  3  |
///       |     |     |     |
/// (0,0) +-----+-----+-----+ (1,0)
auto gen_sprite_quads(size_t count) -> std::tuple<std::vector<TextureVertex>, std::vector<GLushort>>
{
  float width = 1.0f / count;
  std::vector<TextureVertex> vertices;
  std::vector<GLushort> indices;
  vertices.reserve(4 * count);
  indices.reserve(6 * count);
  for (size_t i = 0; i < count; i++) {
    vertices.emplace_back(TextureVertex{ .pos = { +1.0f, +1.0f, +0.0f }, .texcoord = { (i+1)*width, 1.0f } });
    vertices.emplace_back(TextureVertex{ .pos = { +1.0f, -1.0f, +0.0f }, .texcoord = { (i+1)*width, 0.0f } });
    vertices.emplace_back(TextureVertex{ .pos = { -1.0f, -1.0f, +0.0f }, .texcoord = { (i+0)*width, 0.0f } });
    vertices.emplace_back(TextureVertex{ .pos = { -1.0f, +1.0f, +0.0f }, .texcoord = { (i+0)*width, 1.0f } });
    for (auto v : kQuadIndices)
      indices.emplace_back(4*i+v);
  }
  return {vertices, indices};
}

/// Information required to render one frame of a Sprite Animation
struct SpriteFrame {
  float duration;    // duration in seconds, negative is infinite
  size_t ebo_offset; // offset to the first index of this frame in the EBO
  size_t ebo_count;  // number of elements to render since first index
};

/// Control data required for a single Sprite Animation object
struct SpriteAnimation {
  float last_transit_dt; // last transition deltatime 
  size_t curr_frame_idx;   // current frame index
  std::vector<SpriteFrame> frames;

  /// Transition frames
  void update_frame(float dt) {
    last_transit_dt += dt;
    SpriteFrame& curr_frame = frames[curr_frame_idx];
    if (last_transit_dt >= curr_frame.duration) {
      last_transit_dt -= curr_frame.duration;
      if (++curr_frame_idx == frames.size()) curr_frame_idx = 0;
    }
  }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Textures

/// Represents a texture loaded to GPU memory
struct GLTexture {
  UniqueNum<GLuint> id;

  ~GLTexture() {
    if (id) glDeleteTextures(1, &id.inner);
  }

  // Movable but not Copyable
  GLTexture(GLTexture&&) = default;
  GLTexture(const GLTexture&) = delete;
  GLTexture& operator=(GLTexture&&) = default;
  GLTexture& operator=(const GLTexture&) = delete;
};

/// Read file and upload RGB/RBGA texture to GPU memory
auto load_rgba_texture(const std::string& inpath) -> std::optional<GLTexture>
{
  const std::string filepath = SPACESHIP_ASSETS_PATH + "/"s + inpath;
  int width, height, channels;
  stbi_set_flip_vertically_on_load(true);
  unsigned char* data = stbi_load(filepath.data(), &width, &height, &channels, 0);
  if (!data) { ERROR("Failed to load texture path ({})", filepath); return std::nullopt; }
  ASSERT_MSG(channels == 4 || channels == 3, "actual channels: {}", channels);
  GLenum type = (channels == 4) ? GL_RGBA : GL_RGB;
  GLuint texture;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture); 
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, type, width, height, 0, type, GL_UNSIGNED_BYTE, data);
  glGenerateMipmap(GL_TEXTURE_2D);
  stbi_image_free(data);
  return GLTexture{ texture };
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Fonts

/// Upload font bitmap texture to GPU memory
auto load_font_texture(const uint8_t data[], size_t width, size_t height) -> GLTexture
{
  GLuint texture;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture); 
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);	
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, data);
  glGenerateMipmap(GL_TEXTURE_2D);
  return { texture };
}

/// Holds the a font bitmap texture and info needed to render char quads
struct GLFont {
  GLTexture texture;
  int bitmap_px_width;
  int bitmap_px_height;
  int char_beg;
  int char_count;
  std::vector<stbtt_packedchar> chars;
};

/// Read font file and upload generated bitmap texture to GPU memory
auto load_font(const std::string& fontname) -> std::optional<GLFont>
{
  DEBUG("Loading Font {}", fontname);
  const std::string filepath = SPACESHIP_ASSETS_PATH + "/fonts/"s + fontname;
  auto font = read_file_to_string(filepath);
  if (!font) { ERROR("Failed to load font '{}'", fontname); return std::nullopt; }
  constexpr int kStride = 0;
  constexpr int kPadding = 2;
  constexpr unsigned int kOversampling = 2;
  constexpr float kPixelHeight = 22.0;
  constexpr int kCharBeg = 32;
  constexpr int kCharEnd = 128;
  constexpr int kCharCount = kCharEnd - kCharBeg;
  const int kBitmapPixelSize = std::sqrt(kPixelHeight * kPixelHeight * (2.f/3.f) * kCharCount) * kOversampling;
  uint8_t bitmap[kBitmapPixelSize * kBitmapPixelSize];
  stbtt_pack_context pack_ctx;
  stbtt_packedchar packed_chars[kCharCount];
  stbtt_PackBegin(&pack_ctx, bitmap, kBitmapPixelSize, kBitmapPixelSize, kStride, kPadding, nullptr);
  stbtt_PackSetOversampling(&pack_ctx, kOversampling, kOversampling);
  int ret = stbtt_PackFontRange(&pack_ctx, (const uint8_t*)font->data(), 0, STBTT_POINT_SIZE(kPixelHeight), kCharBeg, kCharCount, packed_chars);
  if (ret <= 0) WARN("Font '{}': Some characters may not have fit in the font bitmap!", fontname);
  stbtt_PackEnd(&pack_ctx);
  std::vector<stbtt_packedchar> chars;
  chars.reserve(kCharCount);
  std::copy_n(packed_chars, kCharCount, std::back_inserter(chars));
  GLTexture texture = load_font_texture(bitmap, kBitmapPixelSize, kBitmapPixelSize);
  return GLFont{ 
    .texture = std::move(texture),
    .bitmap_px_width = kBitmapPixelSize,
    .bitmap_px_height = kBitmapPixelSize,
    .char_beg = kCharBeg,
    .char_count = kCharCount,
    .chars = std::move(chars)
  };
}

/// Holds the fonts used by the game
struct Fonts {
  GLFont menlo;
  GLFont jetbrains;
  GLFont google_sans;
  GLFont kanit;
  GLFont russo_one;
};

/// Loads all fonts used by the game
Fonts load_fonts()
{
  return {
    .menlo = ASSERT_OPT(load_font("Menlo-Regular.ttf")),
    .jetbrains = ASSERT_OPT(load_font("JetBrainsMono-Regular.ttf")),
    .google_sans = ASSERT_OPT(load_font("GoogleSans-Regular.ttf")),
    .kanit = ASSERT_OPT(load_font("Kanit/Kanit-Bold.ttf")),
    .russo_one = ASSERT_OPT(load_font("Russo_One/RussoOne-Regular.ttf")),
  };
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Text

/// Generate quad vertices for a text with the given font.
auto gen_text_quads(const GLFont& font, std::string_view text) -> std::tuple<std::vector<TextureVertex>, std::vector<GLushort>>
{
  std::vector<TextureVertex> vertices;
  vertices.reserve(4 * text.size());
  std::vector<GLushort> indices;
  indices.reserve(6 * text.size());
  float x = 0, y = 0;
  size_t i = 0;
  for (char ch : text) {
    if (ch < font.char_beg || ch > (font.char_beg + font.char_count))
      continue;
    stbtt_aligned_quad q;
    stbtt_GetPackedQuad(font.chars.data(), font.bitmap_px_width, font.bitmap_px_height, (int)ch - font.char_beg, &x, &y, &q, 1);
    vertices.emplace_back(TextureVertex{ .pos = { q.x0, q.y0, 0.f }, .texcoord = { q.s0, q.t0 } });
    vertices.emplace_back(TextureVertex{ .pos = { q.x1, q.y0, 0.f }, .texcoord = { q.s1, q.t0 } });
    vertices.emplace_back(TextureVertex{ .pos = { q.x1, q.y1, 0.f }, .texcoord = { q.s1, q.t1 } });
    vertices.emplace_back(TextureVertex{ .pos = { q.x0, q.y1, 0.f }, .texcoord = { q.s0, q.t1 } });
    for (auto v : kQuadIndices)
      indices.emplace_back(4*i+v);
    i++;
  }
  return {vertices, indices};
}

/// Upload new Text Indexed-Vertex object to GPU memory
GLObject create_text_globject(const GLShader& shader, gsl::span<const TextureVertex> vertices, gsl::span<const GLushort> indices, GLenum usage = GL_STATIC_DRAW)
{
  return create_textured_globject(shader, vertices, indices, usage);
}

/// Update GLObject data with the new generated quads for the given Text
auto update_text_globject(const GLShader& shader, GLObject& glo, const GLFont& font, std::string_view text, GLenum usage = GL_STATIC_DRAW)
{
  auto [vertices, indices] = gen_text_quads(font, text);
  if (vertices.size() == glo.num_vertices && indices.size() == glo.num_indices) {
    glBindVertexArray(glo.vao);
    glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.size() * sizeof(TextureVertex), vertices.data());
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, indices.size() * sizeof(GLushort), indices.data());
    glBindVertexArray(0);
  } else {
    glo = create_text_globject(shader, vertices, indices, usage);
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Components

/// Transform component
struct Transform {
  glm::vec3 position;
  glm::vec3 scale;
  glm::quat rotation;

  static Transform identity() {
    return {
      .position = glm::vec3(0.0f),
      .scale = glm::vec3(1.0f),
      .rotation = glm::quat(1.0f, glm::vec3(0.0f)),
    };
  }

  glm::mat4 model_mat() {
    glm::mat4 translation_mat = glm::translate(glm::mat4(1.0f), position);
    glm::mat4 rotation_mat = glm::toMat4(rotation);
    glm::mat4 scale_mat = glm::scale(glm::mat4(1.0f), scale);
    return translation_mat * rotation_mat * scale_mat;
  }
};

/// Motion component
struct Motion {
  glm::vec3 velocity;
  glm::vec3 acceleration;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Renderer

/// Prepare to render
void begin_render()
{
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

/// Upload camera matrix to shader
void set_camera(const GLShader& shader)
{
  auto view = glm::mat4(1.0f);
  auto projection = glm::mat4(1.0f);
  glUniformMatrix4fv(shader.unif_loc(GLUnif::VIEW), 1, GL_FALSE, glm::value_ptr(view));
  glUniformMatrix4fv(shader.unif_loc(GLUnif::PROJECTION), 1, GL_FALSE, glm::value_ptr(projection));
}

/// Render a colored GLObject with indices
void draw_colored_object(const GLShader& shader, const GLObject& glo, const glm::mat4& model)
{
  if (shader.unif_loc(GLUnif::SUBROUTINE) != -1)
    glUniform1i(shader.unif_loc(GLUnif::SUBROUTINE), static_cast<int>(GLSub::COLOR));
  glUniformMatrix4fv(shader.unif_loc(GLUnif::MODEL), 1, GL_FALSE, glm::value_ptr(model));
  glBindVertexArray(glo.vao);
  glDrawElements(GL_TRIANGLES, glo.num_indices, GL_UNSIGNED_SHORT, nullptr);
  glBindVertexArray(0);
}

/// Render a textured GLObject with indices
void draw_textured_object(const GLShader& shader, const GLTexture& texture, const GLObject& glo,
                          const glm::mat4& model, size_t ebo_offset, size_t ebo_count)
{
  if (shader.unif_loc(GLUnif::SUBROUTINE) != -1)
    glUniform1i(shader.unif_loc(GLUnif::SUBROUTINE), static_cast<int>(GLSub::TEXTURE));
  glUniformMatrix4fv(shader.unif_loc(GLUnif::MODEL), 1, GL_FALSE, glm::value_ptr(model));
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texture.id);
  glBindVertexArray(glo.vao);
  glDrawElements(GL_TRIANGLES, ebo_count, GL_UNSIGNED_SHORT, (const void*)ebo_offset);
  glBindVertexArray(0);
}

/// Render a text GLObject with indices
void draw_text_object(const GLShader& shader, const GLTexture& texture, const GLObject& glo,
                      const glm::mat4& model, const glm::vec4& color, const glm::vec4& outline_color, const float outline_thickness)
{
  if (shader.unif_loc(GLUnif::SUBROUTINE) != -1)
    glUniform1i(shader.unif_loc(GLUnif::SUBROUTINE), static_cast<int>(GLSub::FONT));
  glUniform4fv(shader.unif_loc(GLUnif::COLOR), 1, glm::value_ptr(color));
  glUniform4fv(shader.unif_loc(GLUnif::OUTLINE_COLOR), 1, glm::value_ptr(outline_color));
  glUniform1f(shader.unif_loc(GLUnif::OUTLINE_THICKNESS), outline_thickness);
  glUniformMatrix4fv(shader.unif_loc(GLUnif::MODEL), 1, GL_FALSE, glm::value_ptr(model));
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texture.id);
  glBindVertexArray(glo.vao);
  glDrawElements(GL_TRIANGLES, glo.num_indices, GL_UNSIGNED_SHORT, nullptr);
  glBindVertexArray(0);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Game

/// Generic Scene structure
struct Scene {
  struct {
    Transform transform;
    std::optional<GLObject> glo;
    std::optional<GLTexture> texture;
  } background;

  struct {
    Transform transform;
    std::optional<GLObject> glo;
  } colored_quad;

  struct {
    Transform transform;
    std::optional<GLObject> glo;
    std::optional<GLTexture> texture;
  } spaceship;

  struct {
    Transform transform;
    Motion motion;
    std::optional<GLObject> glo;
    std::optional<GLTexture> texture;
    std::optional<SpriteAnimation> animation;
  } ligher;

  struct {
    Transform transform;
    std::optional<GLObject> glo;
  } fps;
};

/// GLFW_KEY_*
using Key = int;
/// Key event handler
using KeyHandler = std::function<void(struct Game&, int key, int action, int mods)>;
/// Map key code to event handlers
using KeyHandlerMap = std::unordered_map<Key, KeyHandler>;
/// Map key to its state, pressed = true, released = false
using KeyStateMap = std::unordered_map<Key, bool>;

/// Game State/Engine
struct Game {
  bool paused;
  GLFWwindow* window;
  std::optional<Shaders> shaders;
  std::optional<Fonts> fonts;
  std::optional<Scene> scene;
  std::optional<KeyHandlerMap> key_handlers;
  std::optional<KeyStateMap> key_states;
};

void init_key_handlers(KeyHandlerMap& key_handlers);

int game_init(Game& game, GLFWwindow* window)
{
  INFO("Initializing game");
  game.paused = false;
  game.window = window;
  game.shaders = load_shaders();
  game.fonts = load_fonts();
  game.scene = Scene{};
  game.key_handlers = KeyHandlerMap(GLFW_KEY_LAST); // reserve all keys to avoid rehash
  game.key_states = KeyStateMap(GLFW_KEY_LAST);     // reserve all keys to avoid rehash

  game.scene->background.transform = Transform::identity();
  game.scene->background.transform.position.z = 0.99f;

  game.scene->colored_quad.transform = Transform{
    .position = glm::vec3(0.45f, 0.0f, -0.1f),
    .scale = glm::vec3(0.3f),
    .rotation = glm::quat(1.0f, glm::vec3(0.0f)),
  };

  game.scene->spaceship.transform = Transform{
    .position = glm::vec3(-0.45f, 0.0f, -0.2f),
    .scale = glm::vec3(0.3f),
    .rotation = glm::quat(1.0f, glm::vec3(0.0f)),
  };

  game.scene->ligher.transform = Transform{
    .position = glm::vec3(0.0f, -0.7f, -0.5f),
    .scale = glm::vec3(0.1f),
    .rotation = glm::quat(1.0f, glm::vec3(0.0f)),
  };

  game.scene->ligher.motion = Motion{
    .velocity = glm::vec3(0.0f),
    .acceleration = glm::vec3(0.0f),
  };

  auto fps_transform = Transform::identity();
  fps_transform.position = glm::vec3(-0.99f);
  fps_transform.scale = glm::vec3(0.0033f);
  fps_transform.scale.y = -fps_transform.scale.y;
  game.scene->fps.transform = std::move(fps_transform);

  DEBUG("Loading Background Texture");
  game.scene->background.texture = load_rgba_texture("background01.png");
  ASSERT(game.scene->background.texture);
  DEBUG("Loading Background Quad");
  game.scene->background.glo = create_textured_quad_globject(game.shaders->generic_shader);

  DEBUG("Loading Colored Quad");
  game.scene->colored_quad.glo = create_colored_quad_globject(game.shaders->generic_shader);

  DEBUG("Loading Spaceship Texture");
  game.scene->spaceship.texture = load_rgba_texture("spaceship.png");
  ASSERT(game.scene->spaceship.texture);
  DEBUG("Loading Spaceship Quad");
  game.scene->spaceship.glo = create_textured_quad_globject(game.shaders->generic_shader);

  DEBUG("Loading Ligher Texture");
  game.scene->ligher.texture = load_rgba_texture("ligher.png");
  ASSERT(game.scene->ligher.texture);
  DEBUG("Loading Ligher Vertices");
  auto [ligher_vertices, ligher_indices] = gen_sprite_quads(4);
  game.scene->ligher.glo = create_textured_globject(game.shaders->generic_shader, ligher_vertices, ligher_indices);
  DEBUG("Loading Ligher Sprite Animation");
  game.scene->ligher.animation = SpriteAnimation{
    .last_transit_dt = 0,
    .curr_frame_idx = 0,
    .frames = std::initializer_list<SpriteFrame>{
      { .duration = 0.1, .ebo_offset = 0, .ebo_count = 6 },
      { .duration = 0.1, .ebo_offset = 6, .ebo_count = 6 },
      { .duration = 0.1, .ebo_offset = 12, .ebo_count = 6 },
      { .duration = 0.1, .ebo_offset = 18, .ebo_count = 6 },
    },
  };

  DEBUG("Loading FPS Text");
  auto [fps_vertices, fps_indices] = gen_text_quads(game.fonts->russo_one, "FPS 00 ms 00.000");
  game.scene->fps.glo = create_text_globject(game.shaders->generic_shader, fps_vertices, fps_indices, GL_DYNAMIC_DRAW);

  return 0;
}

void game_resume(Game& game)
{
  if (!game.paused) return;
  INFO("Resuming game");
  game.paused = false;
}

void game_pause(Game& game)
{
  if (game.paused) return;
  INFO("Pausing game");

  // Release all active keys
  for (auto& key_state : game.key_states.value()) {
    auto& key = key_state.first;
    auto& active = key_state.second;
    if (active) {
      active = false;
      auto it = game.key_handlers->find(key);
      if (it != game.key_handlers->end()) {
        auto& handler = it->second;
        handler(game, key, GLFW_RELEASE, 0);
      }
    }
  }

  game.paused = true;
}

void game_update(Game& game, float dt, float time)
{
  game.scene->colored_quad.transform.position.y = std::sin(time) * -0.4f;
  game.scene->spaceship.transform.position.y = std::sin(time) * 0.4f;
  game.scene->ligher.motion.velocity += game.scene->ligher.motion.acceleration * dt;
  game.scene->ligher.transform.position += game.scene->ligher.motion.velocity * dt;
  game.scene->ligher.animation->update_frame(dt);
}

/// Calculates the average FPS within kPeriod and update FPS GLObject data for render
void update_fps(const GLShader& shader, GLObject& glo, const GLFont& font, float dt)
{
  constexpr float kPeriod = 0.3f; // second
  static size_t counter = 1;
  counter++;
  float fps_now = (1.f / dt);
  static float fps_avg = fps_now;
  fps_avg += fps_now;
  static float fps_dt = 0;
  fps_dt += dt;
  static float fps = fps_avg;
  if (fps_dt > kPeriod) {
    fps_dt -= kPeriod;
    fps = fps_avg / counter;
    fps_avg = fps_now;
    counter = 1;
  }
  static float last_fps = 0;
  if (fps != last_fps) {
    last_fps = fps;
    char fps_cbuf[30];
    float ms = (1.f / fps) * 1000;
    std::snprintf(fps_cbuf, sizeof(fps_cbuf), "FPS %.0f ms %.3f", fps, ms);
    update_text_globject(shader, glo, font, fps_cbuf, GL_DYNAMIC_DRAW);
  }
}

void game_render(Game& game, float dt, float time)
{
  begin_render();

  GLShader& generic_shader = game.shaders->generic_shader;
  generic_shader.bind();
  set_camera(generic_shader);

  auto& colored_quad = game.scene->colored_quad;
  draw_colored_object(generic_shader, *colored_quad.glo, colored_quad.transform.model_mat());

  auto& background = game.scene->background;
  draw_textured_object(generic_shader, *background.texture, *background.glo, background.transform.model_mat(), 0, background.glo->num_indices);

  auto& spaceship = game.scene->spaceship;
  draw_textured_object(generic_shader, *spaceship.texture, *spaceship.glo, spaceship.transform.model_mat(), 0, spaceship.glo->num_indices);

  auto& ligher = game.scene->ligher;
  SpriteFrame& ligher_frame = ligher.animation->frames[ligher.animation->curr_frame_idx];
  draw_textured_object(generic_shader, *ligher.texture, *ligher.glo, ligher.transform.model_mat(), ligher_frame.ebo_offset, ligher_frame.ebo_count);

  auto& fps = game.scene->fps;
  update_fps(generic_shader, *fps.glo, game.fonts->russo_one, dt);
  draw_text_object(generic_shader, game.fonts->russo_one.texture, *fps.glo, fps.transform.model_mat(), glm::vec4(1.f), glm::vec4(glm::vec3(0.f), 1.f), 1.0f);
}

int game_loop(GLFWwindow* window)
{
  Game game;
  int ret = game_init(game, window);
  if (ret) return ret;
  init_key_handlers(*game.key_handlers);
  glfwSetWindowUserPointer(window, &game);
  float last_time = 0;
  while (!glfwWindowShouldClose(window)) {
    float time = glfwGetTime();
    float dt = time - last_time;
    last_time = time;
    game_update(game, dt, time);
    game_render(game, dt, time);
    glfwSwapBuffers(window);
    glfwPollEvents();
  }
  return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Event Handlers

void key_left_right_handler(struct Game& game, int key, int action, int mods)
{
  ASSERT(key == GLFW_KEY_LEFT || key == GLFW_KEY_RIGHT);
  const float kFactor = (key == GLFW_KEY_LEFT ? -1.f : +1.f);
  if (action == GLFW_PRESS || action == GLFW_REPEAT) {
    game.scene->ligher.motion.velocity.x = 0.5f * kFactor;
    game.scene->ligher.motion.acceleration.x = 1.8f * kFactor;
  }
  else if (action == GLFW_RELEASE) {
    const int other_key = (key == GLFW_KEY_LEFT) ? GLFW_KEY_RIGHT : GLFW_KEY_LEFT;
    if (game.key_states.value()[other_key]) {
      // revert to opposite key's movement
      game.key_handlers.value()[other_key](game, other_key, GLFW_REPEAT, mods);
    } else {
      // both arrow keys release, cease movement
      game.scene->ligher.motion.velocity.x = 0.0f;
      game.scene->ligher.motion.acceleration.x = 0.0f;
    }
  }
}

void key_up_down_handler(struct Game& game, int key, int action, int mods)
{
  ASSERT(key == GLFW_KEY_UP || key == GLFW_KEY_DOWN);
  const float kFactor = (key == GLFW_KEY_UP ? +1.f : -1.f);
  if (action == GLFW_PRESS || action == GLFW_REPEAT) {
    game.scene->ligher.motion.velocity.y = 0.5f * kFactor;
    game.scene->ligher.motion.acceleration.y = 1.2f * kFactor;
  }
  else if (action == GLFW_RELEASE) {
    const int other_key = (key == GLFW_KEY_UP) ? GLFW_KEY_DOWN : GLFW_KEY_UP;
    if (game.key_states.value()[other_key]) {
      // revert to opposite key's movement
      game.key_handlers.value()[other_key](game, other_key, GLFW_REPEAT, mods);
    } else {
      // both arrow keys release, cease movement
      game.scene->ligher.motion.velocity.y = 0.0f;
      game.scene->ligher.motion.acceleration.y = 0.0f; 
    }
  }
}

void init_key_handlers(KeyHandlerMap& key_handlers)
{
  key_handlers[GLFW_KEY_LEFT] = key_left_right_handler;
  key_handlers[GLFW_KEY_RIGHT] = key_left_right_handler;
  key_handlers[GLFW_KEY_UP] = key_up_down_handler;
  key_handlers[GLFW_KEY_DOWN] = key_up_down_handler;
}

void key_event_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
  auto game = static_cast<Game*>(glfwGetWindowUserPointer(window));
  if (!game) return; // game not initialized yet

  if (action != GLFW_PRESS && action != GLFW_RELEASE)
    return;

  TRACE("Event key: {} action: {} mods: {}", key, action, mods);

  // Handle Super key (special case)
  if (key == GLFW_KEY_LEFT_SUPER || key == GLFW_KEY_RIGHT_SUPER) {
    if (action == GLFW_PRESS) game_pause(*game);
    else if (action == GLFW_RELEASE) game_resume(*game);
    return;
  }

  // Handle registered keys
  auto it = game->key_handlers->find(key);
  if (it != game->key_handlers->end()) {
    auto& handler = it->second;
    handler(*game, key, action, mods);
  }
  game->key_states.value()[key] = (action == GLFW_PRESS);
}

void window_focus_callback(GLFWwindow* window, int focused)
{
  auto game = static_cast<Game*>(glfwGetWindowUserPointer(window));
  if (!game) return; // game not initialized yet

  if (focused) {
    DEBUG("Window Focused");
    game_resume(*game);
  } else /* unfocused */ {
    DEBUG("Window Unfocused");
    game_pause(*game);
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Setup

int create_window(GLFWwindow*& window)
{
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  window = glfwCreateWindow(500, 700, "Spaceship", nullptr, nullptr);
  if (window == nullptr) {
    std::cerr << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return -3;
  }
  glfwMakeContextCurrent(window);

  // callbacks
  glfwSetKeyCallback(window, key_event_callback);
  glfwSetWindowFocusCallback(window, window_focus_callback);

  // settings
  int width, height;
  glfwGetWindowSize(window, &width, &height);
  glfwSwapInterval(0); // vsync

  return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Main

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
