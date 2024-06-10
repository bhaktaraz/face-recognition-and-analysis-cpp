# Image Recognition Project in C++

## Project Overview

- This C++ project leverages OpenCV, GLFW, and IMGUI to implement various image recognition features. The project includes the following functionalities:

  1. Face Detection
  2. Feature (Eyes, Mouth) Detection
  3. Image Capture
  4. Face Recognition
  5. Facial Emotion Detection


## Prerequisites

Before running the project, ensure you have the following dependencies installed:

1. **C++ Compiler: MinGW**

   - MINGW: https://sourceforge.net/projects/mingw/
   - Add the OpenCV bin directory to the system's environment variables (e.g., C:\opencv\build\x64\vc16\bin).

2. **OpenCV**

   - Download or extract OpenCV to the C:\ directory.
   - Add the OpenCV bin directory to the system's environment variables (e.g., C:\opencv\build\x64\vc16\bin).

3. **CMake**

   - CMake: https://cmake.org/download/

4. **IMGUI and GLFW:**

   - IMGUI and GLFW libraries are included within the project, so there's no need to install them separately.
   - IMGUI: https://github.com/ocornut/imgui
   - GLFW: https://www.glfw.org/

5. **Visual Studio 17 (2022) Community**

   - Download: https://visualstudio.microsoft.com/downloads/
   - Install Visual Studio 2017 or 2022.
   - Ensure to include the "Desktop development with C++" workload during installation.

6. **Visual Studio Code with Necessary Extension**

   - Install Visual Studio Code from: https://code.visualstudio.com/
   - Extensions: C/C++, CMake, CMake Tools,
   - In the CMake configuration in Visual Studio Code, select 'Visual Studio Build Tools 2022 amd64' as the generator and opt for the 'Debug' mode.
   - **Note:** Ensure that you have the necessary C++ extensions installed in Visual Studio Code for a smoother development experience.

7. **Download Other Dependencise CURL and NLOHMANN JSON**
    - To install libcurl and nlohmann/json (json.hpp), follow the steps below based on your development environment.

   - Install libcurl:
   ```
   vcpkg install curl
   ```
   - Install nlohmann/json:
   ```
   vcpkg install nlohmann-json
   ```


## Installing vcpkg on Windows:

vcpkg is a free C/C++ package manager for acquiring and managing libraries.

1. Clone vcpkg:

   - Begin by cloning vcpkg from its GitHub repository:
   - Open a command prompt and run:

   ```
   bash git clone https://github.com/Microsoft/vcpkg.git
   cd vcpkg
   ```

2. Bootstrap vcpkg:

   - Execute the bootstrap script to set up vcpkg:
   - .\bootstrap-vcpkg.bat

3. Integrate vcpkg with Visual Studio:

   - After bootstrapping, seamlessly integrate vcpkg with Visual Studio to make the installed libraries globally available.
   - .\vcpkg integrate install

4. Add vcpkg to PATH (Optional):

   - For convenience, you can add the vcpkg executable to your system's PATH environment variable.

5. Build the project:

   - Proceed to build your project using CMake as usual.
   - you are using VS Code with CMake Tools extension, you might need to configure the .vscode/settings.json for your workspace to include the toolchain file:

   ```
   {
       "cmake.configureSettings": {
           "CMAKE_TOOLCHAIN_FILE": "C:/vcpkg/scripts/buildsystems/vcpkg.cmake"
       }
   }
   ```


## Building and Running the Project

    - Build the Project: Use CMake to generate build files.


# Explore Image Recognition:

The application window will open, showcasing the various image recognition features.
Follow the on-screen instructions to capture images, detect faces, and explore additional functionalities.

Happy coding! 🚀
