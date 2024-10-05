add_executable(01-vector-add)

target_sources(01-vector-add PRIVATE 01-vector-add.cu)

set_target_properties(01-vector-add PROPERTIES
                                        POSITION_INDEPENDENT_CODE ON
                                        CUDA_ARCHITECTURES native
                                        CUDA_SEPARABLE_COMPILATION ON)