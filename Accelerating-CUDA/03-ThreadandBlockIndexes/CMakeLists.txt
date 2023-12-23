add_executable(01-thread-and-block-idx)

target_sources(01-thread-and-block-idx PRIVATE 01-thread-and-block-idx.cu)

set_target_properties(01-thread-and-block-idx PROPERTIES
                                        POSITION_INDEPENDENT_CODE ON
                                        CUDA_SEPARABLE_COMPILATION ON)