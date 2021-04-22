cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pytorch-test)

find_package(Torch REQUIRED)
SET(GCC_DEBUG_FLAGS "-g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} ${GCC_DEBUG_FLAGS}")

add_executable(pytorch-test pytorch-test.cpp ../simulation/components/station.cpp ../simulation/components/queue_graphv2.cpp)
target_link_libraries(pytorch-test "${TORCH_LIBRARIES}")
set_property(TARGET pytorch-test PROPERTY CXX_STANDARD 14)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET pytorch-test
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:pytorch-test>)
endif (MSVC)