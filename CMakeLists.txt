cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-app)

include_directories(external)
include_directories(external/tclap-1.2.2/include)
include_directories(SYSTEM external/filesystem-1.3.6/include) #SYSTEM suppresses a few warnings


set(CMAKE_PREFIX_PATH /Users/syys/Downloads/cpp_torch_app/libtorch)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(example-app main.cpp board.cpp core/global.cpp
        thread_pool.cpp nogo.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 14)

add_executable(thread_pool_test tests/thread_pool_test.cpp)
target_link_libraries(thread_pool_test "${TORCH_LIBRARIES}")
set_property(TARGET thread_pool_test PROPERTY CXX_STANDARD 14)

add_executable(board_test tests/board_test.cpp board.cpp)
target_link_libraries(board_test "${TORCH_LIBRARIES}")
set_property(TARGET board_test PROPERTY CXX_STANDARD 14)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET example-app
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:example-app>)
endif (MSVC)