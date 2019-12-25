find_package(OpenCV)
add_executable(ft_computation
  compute_ft_point.cc
)



target_link_libraries(ft_computation
    ${OpenCV_LIBS}
)