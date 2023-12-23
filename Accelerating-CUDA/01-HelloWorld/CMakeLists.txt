add_executable(01-hello-gpu)

target_sources(01-hello-gpu PRIVATE 01-hello-gpu.cu)

set_target_properties(01-hello-gpu PROPERTIES
                                   POSITION_INDEPENDENT_CODE ON
                                   CUDA_SEPARABLE_COMPILATION ON)