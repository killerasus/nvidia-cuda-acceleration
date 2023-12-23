add_executable(01-matrix-multiply-2d)

target_sources(01-matrix-multiply-2d PRIVATE 01-matrix-multiply-2d.cu)

set_target_properties(01-matrix-multiply-2d PROPERTIES
                                        POSITION_INDEPENDENT_CODE ON
                                        CUDA_SEPARABLE_COMPILATION ON)