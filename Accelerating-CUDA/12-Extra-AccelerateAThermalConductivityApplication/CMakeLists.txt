add_executable(01-heat-conduction)

target_sources(01-heat-conduction PRIVATE 01-heat-conduction.cu)

set_target_properties(01-heat-conduction PROPERTIES
                                        POSITION_INDEPENDENT_CODE ON
                                        CUDA_ARCHITECTURES native
                                        CUDA_SEPARABLE_COMPILATION ON)