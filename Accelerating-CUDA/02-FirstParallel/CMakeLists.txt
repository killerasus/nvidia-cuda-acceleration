add_executable(01-first-parallel)

target_sources(01-first-parallel PRIVATE 01-first-parallel.cu)

set_target_properties(01-first-parallel PROPERTIES
                                        POSITION_INDEPENDENT_CODE ON
                                        CUDA_ARCHITECTURES native
                                        CUDA_SEPARABLE_COMPILATION ON)
