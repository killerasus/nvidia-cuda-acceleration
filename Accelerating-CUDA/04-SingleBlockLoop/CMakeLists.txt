add_executable(01-single-block-loop)

target_sources(01-single-block-loop PRIVATE 01-single-block-loop.cu)

set_target_properties(01-single-block-loop PROPERTIES
                                            POSITION_INDEPENDENT_CODE ON
                                            CUDA_ARCHITECTURES native
                                            CUDA_SEPARABLE_COMPILATION ON)