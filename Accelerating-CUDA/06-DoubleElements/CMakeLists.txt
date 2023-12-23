add_executable(01-double-elements)

target_sources(01-double-elements PRIVATE 01-double-elements.cu)

set_target_properties(01-double-elements PROPERTIES
                                        POSITION_INDEPENDENT_CODE ON
                                        CUDA_SEPARABLE_COMPILATION ON)