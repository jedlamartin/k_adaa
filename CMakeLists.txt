cmake_minimum_required(VERSION 3.14)
project(KMethodProcessor CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PAPI REQUIRED papi)

add_executable(audio_processor
    main.cpp
    K_method.cpp
)

target_link_libraries(audio_processor PRIVATE Eigen3::Eigen)

target_include_directories(audio_processor PRIVATE ${PAPI_INCLUDE_DIRS})
target_link_libraries(audio_processor PRIVATE ${PAPI_LIBRARIES} pthread dl)