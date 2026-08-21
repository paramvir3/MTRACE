#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [--symlink] /path/to/lammps" >&2
  exit 2
}

mode=copy
if [[ ${1:-} == "--symlink" ]]; then
  mode=symlink
  shift
fi
[[ $# -eq 1 ]] || usage

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
lammps_dir=$(cd "$1" && pwd)
[[ -f "$lammps_dir/cmake/CMakeLists.txt" && -d "$lammps_dir/src/KOKKOS" ]] || {
  echo "Not a LAMMPS source tree: $lammps_dir" >&2
  exit 1
}

install_file() {
  local source=$1 destination=$2
  if [[ $mode == symlink ]]; then
    ln -sfn "$source" "$destination"
  else
    cp "$source" "$destination"
  fi
}

install_file "$script_dir/pair_mamba.cpp" "$lammps_dir/src/pair_mamba.cpp"
install_file "$script_dir/pair_mamba.h" "$lammps_dir/src/pair_mamba.h"
install_file "$script_dir/pair_mamba_kokkos.cpp" "$lammps_dir/src/KOKKOS/pair_mamba_kokkos.cpp"
install_file "$script_dir/pair_mamba_kokkos.h" "$lammps_dir/src/KOKKOS/pair_mamba_kokkos.h"
install_file "$script_dir/cmake/MAMBAACE.cmake" "$lammps_dir/cmake/Modules/MAMBAACE.cmake"

marker="# MAMBA_ACE_LIBTORCH"
if ! grep -qF "$marker" "$lammps_dir/cmake/CMakeLists.txt"; then
  printf '\n%s\ninclude(Modules/MAMBAACE.cmake)\n' "$marker" >> "$lammps_dir/cmake/CMakeLists.txt"
fi

echo "MAMBA-ACE pair styles installed in $lammps_dir"
echo "Configure CMake with -D CMAKE_PREFIX_PATH=\$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
