add_executable(03-grid-stride-double)

target_sources(03-grid-stride-double PRIVATE 03-grid-stride-double.cu)

set_target_properties(03-grid-stride-double PROPERTIES
                                        POSITION_INDEPENDENT_CODE ON
                                        CUDA_SEPARABLE_COMPILATION ON)