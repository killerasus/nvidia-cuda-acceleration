add_executable(02-multi-block-loop)

target_sources(02-multi-block-loop PRIVATE 02-multi-block-loop.cu)

set_target_properties(02-multi-block-loop PROPERTIES
                                        POSITION_INDEPENDENT_CODE ON
                                        CUDA_ARCHITECTURES native
                                        CUDA_SEPARABLE_COMPILATION ON)