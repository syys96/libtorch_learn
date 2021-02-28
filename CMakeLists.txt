cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-app)

configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
        ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
        EXCLUDE_FROM_ALL)

# The gtest/gtest_main targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if (CMAKE_VERSION VERSION_LESS 2.8.11)
    include_directories("${gtest_SOURCE_DIR}/include")
endif()

include_directories(external)
include_directories(external/tclap-1.2.2/include)
include_directories(SYSTEM external/filesystem-1.3.6/include) #SYSTEM suppresses a few warnings

set(CMAKE_PREFIX_PATH /Users/syys/Downloads/cpp_torch_app/libtorch)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_definitions(-DUSE_CUDA=0)

add_executable(example-app main.cpp board.cpp core/global.cpp
        thread_pool.cpp nogo.cpp botzone/parase_input.cpp libtorch.cpp
        policy_value_net.cpp mcts.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 14)

add_executable(self_play_test tests/self_play_test.cpp board.cpp core/global.cpp
        thread_pool.cpp nogo.cpp libtorch.cpp policy_value_net.cpp mcts.cpp)
target_link_libraries(self_play_test "${TORCH_LIBRARIES}")
set_property(TARGET self_play_test PROPERTY CXX_STANDARD 14)

add_executable(runtool runtool.cpp board.cpp core/global.cpp
        thread_pool.cpp nogo.cpp botzone/parase_input.cpp libtorch.cpp
        policy_value_net.cpp mcts.cpp)
target_link_libraries(runtool "${TORCH_LIBRARIES}")
set_property(TARGET runtool PROPERTY CXX_STANDARD 14)

add_executable(thread_pool_test tests/thread_pool_test.cpp)
target_link_libraries(thread_pool_test "${TORCH_LIBRARIES}")
set_property(TARGET thread_pool_test PROPERTY CXX_STANDARD 14)

add_executable(board_test tests/board_test.cpp board.cpp)
target_link_libraries(board_test "${TORCH_LIBRARIES}" gtest_main)
set_property(TARGET board_test PROPERTY CXX_STANDARD 14)
enable_testing()

add_executable(cpp_net_test tests/policy_value_net_test.cpp)
target_link_libraries(cpp_net_test "${TORCH_LIBRARIES}" gtest_main)
set_property(TARGET cpp_net_test PROPERTY CXX_STANDARD 14)
enable_testing()

set(CMAKE_CXX_STANDARD 14)
add_executable(gtest_demo tests/unittest.cpp)
target_link_libraries(gtest_demo gtest_main)
enable_testing()

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