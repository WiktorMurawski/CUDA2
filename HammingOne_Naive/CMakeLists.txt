add_executable(HammingOne_Naive "main.cu")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET HammingOne_Naive PROPERTY CXX_STANDARD 20)
endif()

# Modern way: Use CUDAToolkit imported targets
find_package(CUDAToolkit REQUIRED)

target_link_libraries(HammingOne_Naive PRIVATE CUDA::cudart)  # Links runtime, adds includes automatically
# Optional: CUDA::cuda_driver or others if needed

add_custom_command(TARGET HammingOne_Naive POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/Tests
        $<TARGET_FILE_DIR:HammingOne_Naive>/Tests
    COMMENT "Copying Tests folder to executable directory"
)