cmake_minimum_required(VERSION 3.16)
project(immed1ate_seal LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# SEAL_DIR: the directory holding SEALConfig.cmake (seal_backend/build.sh passes seal_backend/_deps/install/...)
find_package(SEAL 4.1 REQUIRED)

add_library(imm_core src/seal_core.cpp src/io.cpp)
target_include_directories(imm_core PUBLIC include third_party)
target_link_libraries(imm_core PUBLIC SEAL::seal)

foreach(p imm copse vesta)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/${p}_bench.cpp)
    add_executable(${p}_bench src/${p}_bench.cpp)
    target_link_libraries(${p}_bench PRIVATE imm_core SEAL::seal)
  endif()
endforeach()

add_executable(core_test tests/core_test.cpp)
target_link_libraries(core_test PRIVATE imm_core SEAL::seal)
enable_testing()
add_test(NAME core COMMAND core_test)
