cmake_minimum_required(VERSION 3.10)

# Set the toolchain file before calling project()
# Set the vcpkg toolchain file if it's not already set
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
set(CMAKE_TOOLCHAIN_FILE "C:/Users/Bibek Joshi/Desktop/vcpkg/scripts/buildsystems/vcpkg.cmake"
CACHE STRING "Vcpkg toolchain file")
endif()

project(ImageRecognition)

# Find OpenCV
set(OPENCV_DIR C:/opencv/build)
find_package(OpenCV REQUIRED)

# Find nlohmann_json
set(nlohmann_json_DIR "C:/Users/Bibek Joshi/Desktop/vcpkg/installed/x64-windows/share/nlohmann_json")
find_package(nlohmann_json CONFIG REQUIRED)

# Find CURL
set(CURL_INCLUDE_DIR "C:/Users/Bibek Joshi/Desktop/vcpkg/installed/x64-windows/share/curl")
find_package(CURL REQUIRED)

# Add ImGui and its backends
add_subdirectory(imgui)

# Add the executable
add_executable(${PROJECT_NAME} src/main.cpp)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(glfw)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
  ${OpenCV_LIBS}
  imgui
  glfw
  ${GLFW_LIBRARIES}
  OpenGL32.lib
  CURL::libcurl
  nlohmann_json::nlohmann_json
)
