cmake_minimum_required(VERSION 3.27)
project( ml_training )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_subdirectory(knn)

add_executable( main main.cpp )

target_link_libraries( main PUBLIC ${OpenCV_LIBS} knn)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(knn PUBLIC OpenMP::OpenMP_CXX)
endif()

target_include_directories(main PUBLIC "./knn")
set_property(TARGET main PROPERTY CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -O2")
