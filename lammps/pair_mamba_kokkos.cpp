/* ----------------------------------------------------------------------
   Zero-copy Kokkos/LibTorch device path for MAMBA-ACE.
------------------------------------------------------------------------- */

#include "pair_mamba_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "memory_kokkos.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"

#include <torch/cuda.h>
#include <c10/core/DeviceGuard.h>

#include <algorithm>
#include <type_traits>

#ifndef LMP_KOKKOS_DOUBLE_DOUBLE
#error "Pair style mamba/kk requires a double/double LAMMPS Kokkos build"
#endif

using namespace LAMMPS_NS;

template <class DeviceType>
PairMAMBAKokkos<DeviceType>::PairMAMBAKokkos(LAMMPS *lmp) : PairMAMBA(lmp)
{
  atomKK = dynamic_cast<AtomKokkos *>(atom);
  if (atomKK == nullptr) error->all(FLERR, "Pair style mamba/kk requires Kokkos atoms");
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | TYPE_MASK | ENERGY_MASK;
  datamask_modify = F_MASK | ENERGY_MASK;
  kokkosable = 1;
  reverse_comm_device = 1;
}

template <class DeviceType> PairMAMBAKokkos<DeviceType>::~PairMAMBAKokkos()
{
  if (!copymode) {
    memoryKK->destroy_kokkos(k_eatom, eatom);
    eatom = nullptr;
  }
}

template <class DeviceType> void PairMAMBAKokkos<DeviceType>::coeff(int narg, char **arg)
{
#if !defined(KOKKOS_ENABLE_CUDA)
  if (requested_device == "auto") requested_device = "cpu";
#endif
  PairMAMBA::coeff(narg, arg);
#if defined(KOKKOS_ENABLE_CUDA)
  if (!device.is_cuda())
    error->all(FLERR, "A CUDA Kokkos build requires pair_style mamba/kk device cuda");
  if (device.index() != Kokkos::Cuda().cuda_device())
    error->all(FLERR,
               "MAMBA-ACE selected CUDA device {}, but Kokkos uses device {}; bind one MPI rank per GPU",
               device.index(), Kokkos::Cuda().cuda_device());
#elif defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
  error->all(FLERR, "Pair style mamba/kk currently supports Kokkos host and CUDA backends");
#else
  if (!device.is_cpu())
    error->all(FLERR, "A host Kokkos build requires pair_style mamba/kk device cpu");
#endif

  type_to_z = Int64View("mamba:type_to_z", type_to_atomic_number.size());
  auto host = Kokkos::create_mirror_view(type_to_z);
  for (size_t index = 0; index < type_to_atomic_number.size(); ++index)
    host(index) = type_to_atomic_number[index];
  Kokkos::deep_copy(type_to_z, host);
}

template <class DeviceType> void PairMAMBAKokkos<DeviceType>::init_style()
{
  PairMAMBA::init_style();
  auto *request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same<DeviceType, LMPHostType>::value &&
                           !std::is_same<DeviceType, LMPDeviceType>::value);
  request->set_kokkos_device(std::is_same<DeviceType, LMPDeviceType>::value);
  neighflag = lmp->kokkos->neighflag;
  if (neighflag == FULL)
    error->all(FLERR, "Pair style mamba/kk requires 'package kokkos neigh half' with newton on");
}

template <class DeviceType> void PairMAMBAKokkos<DeviceType>::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag, 0);
  if (vflag_atom || cvflag_atom)
    error->all(FLERR, "Pair style mamba/kk does not define a per-atom virial");

  if (eflag_atom) {
    memoryKK->destroy_kokkos(k_eatom, eatom);
    memoryKK->create_kokkos(k_eatom, eatom, maxeatom, "pair_mamba:eatom");
    d_eatom = k_eatom.template view<DeviceType>();
  }

  atomKK->sync(execution_space, datamask_read);
  if (eflag || vflag)
    atomKK->modified(execution_space, datamask_modify);
  else
    atomKK->modified(execution_space, F_MASK);
  x = atomKK->k_x.template view<DeviceType>();
  f = atomKK->k_f.template view<DeviceType>();
  type = atomKK->k_type.template view<DeviceType>();

  const int nlocal = atom->nlocal;
  const int nall = nlocal + atom->nghost;
  if (nlocal == 0) return;
  if (list->inum != nlocal)
    error->all(FLERR, "Pair style mamba/kk requires all owned atoms in its neighbor list");

  auto *kokkos_list = static_cast<NeighListKokkos<DeviceType> *>(list);
  ilist = kokkos_list->d_ilist;
  numneigh = kokkos_list->d_numneigh;
  neighbors = kokkos_list->d_neighbors;

  const bool use_float = model_dtype == torch::kFloat32;
  const int64_t atom_capacity = std::max<int64_t>(1, nall + nall / 8 + 8);
  if (atomic_numbers.extent(0) < static_cast<size_t>(nall)) {
    atomic_numbers = Int64View("mamba:atomic_numbers", atom_capacity);
    if (use_float) {
      positions_float = FloatView2D("mamba:positions_float", atom_capacity, 3);
      positions_double = DoubleView2D();
    } else {
      positions_double = DoubleView2D("mamba:positions_double", atom_capacity, 3);
      positions_float = FloatView2D();
    }
  }
  if (edge_counts.extent(0) < static_cast<size_t>(nlocal)) {
    edge_counts = IntView("mamba:edge_counts", nlocal + nlocal / 8 + 8);
    edge_offsets = Int64View("mamba:edge_offsets", nlocal + nlocal / 8 + 9);
  }

  auto d_x = x;
  auto d_type = type;
  auto d_type_to_z = type_to_z;
  auto d_numbers = atomic_numbers;
  auto d_pos_float = positions_float;
  auto d_pos_double = positions_double;
  const double origin_x = 0.5 * (domain->boxlo[0] + domain->boxhi[0]);
  const double origin_y = 0.5 * (domain->boxlo[1] + domain->boxhi[1]);
  const double origin_z = 0.5 * (domain->boxlo[2] + domain->boxhi[2]);
  Kokkos::parallel_for(
      "mamba:pack_atoms", Kokkos::RangePolicy<DeviceType>(0, nall), KOKKOS_LAMBDA(const int i) {
        d_numbers(i) = d_type_to_z(d_type(i));
        if (use_float) {
          d_pos_float(i, 0) = static_cast<float>(d_x(i, 0) - origin_x);
          d_pos_float(i, 1) = static_cast<float>(d_x(i, 1) - origin_y);
          d_pos_float(i, 2) = static_cast<float>(d_x(i, 2) - origin_z);
        } else {
          d_pos_double(i, 0) = d_x(i, 0) - origin_x;
          d_pos_double(i, 1) = d_x(i, 1) - origin_y;
          d_pos_double(i, 2) = d_x(i, 2) - origin_z;
        }
      });

  auto d_ilist = ilist;
  auto d_numneigh = numneigh;
  auto d_neighbors = neighbors;
  auto d_counts = edge_counts;
  const double cutoff_squared_double = cutoff * cutoff;
  const float cutoff_float = static_cast<float>(cutoff);
  const float cutoff_squared_float = cutoff_float * cutoff_float;
  Kokkos::parallel_for(
      "mamba:count_edges", Kokkos::RangePolicy<DeviceType>(0, nlocal), KOKKOS_LAMBDA(const int ii) {
        const int i = d_ilist(ii);
        int count = 0;
        for (int jj = 0; jj < d_numneigh(i); ++jj) {
          const int j = d_neighbors(i, jj) & NEIGHMASK;
          bool inside;
          if (use_float) {
            const float dx = d_pos_float(j, 0) - d_pos_float(i, 0);
            const float dy = d_pos_float(j, 1) - d_pos_float(i, 1);
            const float dz = d_pos_float(j, 2) - d_pos_float(i, 2);
            inside = dx * dx + dy * dy + dz * dz < cutoff_squared_float;
          } else {
            const double dx = d_pos_double(j, 0) - d_pos_double(i, 0);
            const double dy = d_pos_double(j, 1) - d_pos_double(i, 1);
            const double dz = d_pos_double(j, 2) - d_pos_double(i, 2);
            inside = dx * dx + dy * dy + dz * dz < cutoff_squared_double;
          }
          if (inside) ++count;
        }
        d_counts(ii) = count;
      });

  auto d_offsets = edge_offsets;
  Kokkos::parallel_scan(
      "mamba:edge_offsets", Kokkos::RangePolicy<DeviceType>(0, nlocal + 1),
      KOKKOS_LAMBDA(const int ii, int64_t &update, const bool final) {
        if (final) d_offsets(ii) = update;
        if (ii < nlocal) update += d_counts(ii);
      });
  int64_t nedges = 0;
  Kokkos::deep_copy(nedges, Kokkos::subview(edge_offsets, nlocal));

  const int64_t edge_capacity = std::max<int64_t>(1, nedges + nedges / 8 + 8);
  if (edges.extent(1) < static_cast<size_t>(nedges))
    edges = Int64View2D("mamba:edges", 2, edge_capacity);
  auto d_edges = edges;
  Kokkos::parallel_for(
      "mamba:pack_edges", Kokkos::RangePolicy<DeviceType>(0, nlocal), KOKKOS_LAMBDA(const int ii) {
        const int i = d_ilist(ii);
        int64_t index = d_offsets(ii);
        for (int jj = 0; jj < d_numneigh(i); ++jj) {
          const int j = d_neighbors(i, jj) & NEIGHMASK;
          bool inside;
          if (use_float) {
            const float dx = d_pos_float(j, 0) - d_pos_float(i, 0);
            const float dy = d_pos_float(j, 1) - d_pos_float(i, 1);
            const float dz = d_pos_float(j, 2) - d_pos_float(i, 2);
            inside = dx * dx + dy * dy + dz * dz < cutoff_squared_float;
          } else {
            const double dx = d_pos_double(j, 0) - d_pos_double(i, 0);
            const double dy = d_pos_double(j, 1) - d_pos_double(i, 1);
            const double dz = d_pos_double(j, 2) - d_pos_double(i, 2);
            inside = dx * dx + dy * dy + dz * dz < cutoff_squared_double;
          }
          if (!inside) continue;
          d_edges(0, index) = j;
          d_edges(1, index) = i;
          ++index;
        }
      });
  Kokkos::fence("mamba:Kokkos-to-LibTorch");

  try {
    c10::DeviceGuard device_guard(device);
    auto tensor_device = device;
    auto z_tensor = torch::from_blob(
        atomic_numbers.data(), {nall},
        torch::TensorOptions().dtype(torch::kInt64).device(tensor_device));
    auto edge_tensor = torch::from_blob(
        edges.data(), {2, nedges}, {static_cast<int64_t>(edges.extent(1)), 1},
        torch::TensorOptions().dtype(torch::kInt64).device(tensor_device));
    torch::Tensor position_tensor;
    if (use_float) {
      position_tensor = torch::from_blob(
          positions_float.data(), {nall, 3}, {3, 1},
          torch::TensorOptions().dtype(torch::kFloat32).device(tensor_device));
    } else {
      position_tensor = torch::from_blob(
          positions_double.data(), {nall, 3}, {3, 1},
          torch::TensorOptions().dtype(torch::kFloat64).device(tensor_device));
    }

    auto result = evaluate(z_tensor, position_tensor, edge_tensor, nlocal, eflag_global != 0,
                           vflag_global != 0);
    auto force_tensor = result.forces.contiguous();
    torch::Tensor energy_tensor;
    if (eflag_atom) energy_tensor = result.atomic_energy.contiguous();
#if defined(KOKKOS_ENABLE_CUDA)
    torch::cuda::synchronize(device.index());
#endif

    auto d_f = f;
    auto d_atomic_energy = d_eatom;
    const bool store_atomic_energy = eflag_atom != 0;
    if (use_float) {
      using View = Kokkos::View<float **, Kokkos::LayoutRight, DeviceType,
                                Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
      View force_values(force_tensor.template data_ptr<float>(), nall, 3);
      Kokkos::parallel_for(
          "mamba:accumulate_forces", Kokkos::RangePolicy<DeviceType>(0, nall),
          KOKKOS_LAMBDA(const int i) {
            d_f(i, 0) += force_values(i, 0);
            d_f(i, 1) += force_values(i, 1);
            d_f(i, 2) += force_values(i, 2);
          });
      if (store_atomic_energy) {
        using EnergyView = Kokkos::View<float *, Kokkos::LayoutRight, DeviceType,
                                        Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        EnergyView energy_values(energy_tensor.template data_ptr<float>(), nall);
        Kokkos::parallel_for(
            "mamba:store_atomic_energy", Kokkos::RangePolicy<DeviceType>(0, nlocal),
            KOKKOS_LAMBDA(const int i) { d_atomic_energy(i) = energy_values(i); });
      }
    } else {
      using View = Kokkos::View<double **, Kokkos::LayoutRight, DeviceType,
                                Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
      View force_values(force_tensor.template data_ptr<double>(), nall, 3);
      Kokkos::parallel_for(
          "mamba:accumulate_forces", Kokkos::RangePolicy<DeviceType>(0, nall),
          KOKKOS_LAMBDA(const int i) {
            d_f(i, 0) += force_values(i, 0);
            d_f(i, 1) += force_values(i, 1);
            d_f(i, 2) += force_values(i, 2);
          });
      if (store_atomic_energy) {
        using EnergyView = Kokkos::View<double *, Kokkos::LayoutRight, DeviceType,
                                        Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        EnergyView energy_values(energy_tensor.template data_ptr<double>(), nall);
        Kokkos::parallel_for(
            "mamba:store_atomic_energy", Kokkos::RangePolicy<DeviceType>(0, nlocal),
            KOKKOS_LAMBDA(const int i) { d_atomic_energy(i) = energy_values(i); });
      }
    }
    Kokkos::fence("mamba:LibTorch-to-Kokkos");
    if (eflag_atom) {
      k_eatom.template modify<DeviceType>();
      k_eatom.template sync<LMPHostType>();
    }
    if (eflag_global) eng_vdwl = result.total_energy;
    if (vflag_global) {
      auto virial_cpu = result.virial.to(torch::kCPU, torch::kFloat64).contiguous();
      auto value = virial_cpu.template accessor<double, 2>();
      virial[0] = value[0][0];
      virial[1] = value[1][1];
      virial[2] = value[2][2];
      virial[3] = value[0][1];
      virial[4] = value[0][2];
      virial[5] = value[1][2];
    }
  } catch (const c10::Error &exception) {
    error->all(FLERR, "MAMBA-ACE Kokkos/LibTorch execution failed: {}", exception.what());
  }
}

namespace LAMMPS_NS {
template class PairMAMBAKokkos<LMPDeviceType>;
}
