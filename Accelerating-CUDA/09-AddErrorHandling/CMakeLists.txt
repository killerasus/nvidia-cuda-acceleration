add_executable(01-add-error-handling)

target_sources(01-add-error-handling PRIVATE 01-add-error-handling.cu)

set_target_properties(01-add-error-handling PROPERTIES
                                        POSITION_INDEPENDENT_CODE ON
                                        CUDA_SEPARABLE_COMPILATION ON)