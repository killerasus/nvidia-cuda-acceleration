add_executable(02-mismatched-config-loop)

target_sources(02-mismatched-config-loop PRIVATE 02-mismatched-config-loop.cu)

set_target_properties(02-mismatched-config-loop PROPERTIES
                                        POSITION_INDEPENDENT_CODE ON
                                        CUDA_ARCHITECTURES native
                                        CUDA_SEPARABLE_COMPILATION ON)