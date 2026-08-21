# LibTorch integration for the MAMBA-ACE pair styles.
find_package(Torch 2.2 REQUIRED)
separate_arguments(MAMBA_ACE_TORCH_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(lammps PRIVATE ${MAMBA_ACE_TORCH_FLAGS})
target_link_libraries(lammps PUBLIC ${TORCH_LIBRARIES})
target_compile_features(lammps PUBLIC cxx_std_17)

message(STATUS "MAMBA-ACE: LibTorch ${Torch_VERSION}")
