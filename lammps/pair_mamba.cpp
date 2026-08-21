/* ----------------------------------------------------------------------
   MAMBA-ACE pair style for LAMMPS.

   Each rank evaluates atomic energies for owned centers only. Gradients are
   taken with respect to owned and ghost coordinates; LAMMPS reverse force
   communication then assembles cross-domain force contributions.
------------------------------------------------------------------------- */

#include "pair_mamba.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"
#include "utils.h"

#include <ATen/Parallel.h>
#include <c10/core/DeviceGuard.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

using namespace LAMMPS_NS;

namespace {

std::vector<std::string> split_words(const std::string &value)
{
  std::istringstream stream(value);
  std::vector<std::string> result;
  for (std::string word; stream >> word;) result.push_back(word);
  return result;
}

std::vector<int64_t> split_integers(const std::string &value)
{
  std::istringstream stream(value);
  std::vector<int64_t> result;
  for (int64_t number; stream >> number;) result.push_back(number);
  return result;
}

bool parse_boolean(const char *value)
{
  if (strcmp(value, "yes") == 0 || strcmp(value, "on") == 0 || strcmp(value, "true") == 0)
    return true;
  if (strcmp(value, "no") == 0 || strcmp(value, "off") == 0 || strcmp(value, "false") == 0)
    return false;
  throw std::invalid_argument("expected yes/no");
}

}    // namespace

PairMAMBA::PairMAMBA(LAMMPS *lmp) : Pair(lmp)
{
  restartinfo = 0;
  respa_enable = 0;
  manybody_flag = 1;
  one_coeff = 1;
  single_enable = 0;
  no_virial_fdotr_compute = 1;
}

PairMAMBA::~PairMAMBA()
{
  if (copymode) return;
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

void PairMAMBA::allocate()
{
  allocated = 1;
  const int n = atom->ntypes + 1;
  memory->create(setflag, n, n, "pair_mamba:setflag");
  memory->create(cutsq, n, n, "pair_mamba:cutsq");
}

void PairMAMBA::settings(int narg, char **arg)
{
  if (strcmp("metal", update->unit_style) != 0)
    error->all(FLERR, "Pair style mamba requires 'units metal'");

  int index = 0;
  while (index < narg) {
    if (strcmp(arg[index], "device") == 0) {
      if (index + 1 >= narg) utils::missing_cmd_args(FLERR, "pair_style mamba device", error);
      requested_device = arg[index + 1];
      index += 2;
    } else if (strcmp(arg[index], "threads") == 0) {
      if (index + 1 >= narg) utils::missing_cmd_args(FLERR, "pair_style mamba threads", error);
      intraop_threads = utils::inumeric(FLERR, arg[index + 1], false, lmp);
      if (intraop_threads < 1) error->all(FLERR, "Pair style mamba threads must be positive");
      index += 2;
    } else if (strcmp(arg[index], "debug") == 0) {
      if (index + 1 >= narg) utils::missing_cmd_args(FLERR, "pair_style mamba debug", error);
      try {
        debug = parse_boolean(arg[index + 1]);
      } catch (const std::invalid_argument &) {
        error->all(FLERR, "Pair style mamba debug expects yes or no");
      }
      index += 2;
    } else if (strcmp(arg[index], "check_finite") == 0) {
      if (index + 1 >= narg)
        utils::missing_cmd_args(FLERR, "pair_style mamba check_finite", error);
      try {
        check_finite = parse_boolean(arg[index + 1]);
      } catch (const std::invalid_argument &) {
        error->all(FLERR, "Pair style mamba check_finite expects yes or no");
      }
      index += 2;
    } else {
      error->all(FLERR, "Unknown pair_style mamba keyword: {}", arg[index]);
    }
  }
}

void PairMAMBA::select_device()
{
  std::string choice = requested_device;
  if (choice == "auto") {
    if (const char *environment = std::getenv("MAMBA_ACE_DEVICE")) choice = environment;
  }
  if (choice == "auto") choice = torch::cuda::is_available() ? "cuda" : "cpu";

  if (choice == "cpu") {
    device = torch::Device(torch::kCPU);
    return;
  }
  if (choice != "cuda" && choice.rfind("cuda:", 0) != 0)
    error->all(FLERR, "Pair style mamba device must be auto, cpu, cuda, or cuda:N");
  if (!torch::cuda::is_available())
    error->all(FLERR, "Pair style mamba requested CUDA but LibTorch reports no CUDA device");
  const int count = static_cast<int>(torch::cuda::device_count());
  if (count < 1)
    error->all(FLERR, "Pair style mamba requested CUDA but no CUDA devices are visible");

  int device_index = -1;
  if (choice.rfind("cuda:", 0) == 0) {
    try {
      const std::string index_text = choice.substr(5);
      size_t parsed = 0;
      device_index = std::stoi(index_text, &parsed);
      if (parsed != index_text.size()) throw std::invalid_argument("trailing characters");
    } catch (const std::exception &) {
      error->all(FLERR, "Invalid pair_style mamba CUDA device: {}", choice);
    }
  } else {
#if defined(MPI_STUBS)
    // Serial LAMMPS. The bundled MPI stubs implement neither
    // MPI_Comm_split_type nor MPI_COMM_TYPE_SHARED, and there is exactly one
    // rank, so the shared-communicator rank that would select the device is
    // identically zero. Guarding on MPI_STUBS is how LAMMPS itself detects a
    // stub build (see platform::mpi_vendor).
    device_index = 0;
#else
    // One rank per GPU, mapped by the rank index within each shared-memory
    // node so that ranks on the same host do not collide on device 0.
    MPI_Comm shared;
    MPI_Comm_split_type(world, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shared);
    MPI_Comm_rank(shared, &device_index);
    MPI_Comm_free(&shared);
    if (count == 1) device_index = 0;
#endif
  }
  if (device_index < 0 || device_index >= count)
    error->all(FLERR,
               "Pair style mamba maps this MPI rank to CUDA device {}, but only {} devices are visible",
               device_index, count);
  device = torch::Device(torch::kCUDA, device_index);
}

void PairMAMBA::load_model(const std::string &path)
{
  model_loaded = false;
  std::unordered_map<std::string, std::string> metadata = {
      {"mamba_ace_format_version", ""}, {"architecture", ""},
      {"architecture_version", ""},    {"r_max", ""},
      {"elements", ""},               {"atomic_numbers", ""},
      {"dtype", ""},                  {"energy_units", ""},
      {"length_units", ""},           {"output", ""},
      {"strictly_local", ""}};
  try {
    c10::DeviceGuard device_guard(device);
    model = torch::jit::load(path, device, metadata);
    model.eval();
  } catch (const c10::Error &exception) {
    error->all(FLERR, "Could not load MAMBA-ACE model '{}': {}", path, exception.what());
  }
  if (metadata["mamba_ace_format_version"] != "1")
    error->all(FLERR, "Unsupported or missing MAMBA-ACE deployment format in '{}'", path);
  if (metadata["strictly_local"] != "1" || metadata["output"] != "atomic_energy")
    error->all(FLERR, "Model '{}' does not satisfy the local atomic-energy ABI", path);
  if (metadata["energy_units"] != "eV" || metadata["length_units"] != "Angstrom")
    error->all(FLERR, "Model '{}' is not expressed in eV and Angstrom", path);
  try {
    size_t parsed = 0;
    cutoff = std::stod(metadata["r_max"], &parsed);
    if (parsed != metadata["r_max"].size()) throw std::invalid_argument("trailing characters");
  } catch (const std::exception &) {
    error->all(FLERR, "Model '{}' has invalid r_max metadata", path);
  }
  if (!(cutoff > 0.0) || !std::isfinite(cutoff))
    error->all(FLERR, "Model '{}' has a nonpositive or nonfinite cutoff", path);
  model_elements = split_words(metadata["elements"]);
  model_atomic_numbers = split_integers(metadata["atomic_numbers"]);
  if (model_elements.empty() || model_elements.size() != model_atomic_numbers.size())
    error->all(FLERR, "Model '{}' has inconsistent element metadata", path);
  for (size_t index = 0; index < model_atomic_numbers.size(); ++index) {
    const int64_t number = model_atomic_numbers[index];
    if (number < 1 || number > 118)
      error->all(FLERR, "Model '{}' contains invalid atomic number {}", path, number);
    if (std::count(model_atomic_numbers.begin(), model_atomic_numbers.end(), number) != 1)
      error->all(FLERR, "Model '{}' contains duplicate atomic number {}", path, number);
    if (std::count(model_elements.begin(), model_elements.end(), model_elements[index]) != 1)
      error->all(FLERR, "Model '{}' contains duplicate element name '{}'", path,
                 model_elements[index]);
  }
  if (metadata["dtype"] == "float32")
    model_dtype = torch::kFloat32;
  else if (metadata["dtype"] == "float64")
    model_dtype = torch::kFloat64;
  else
    error->all(FLERR, "Model '{}' has unsupported dtype '{}'", path, metadata["dtype"]);
  model_loaded = true;
}

void PairMAMBA::coeff(int narg, char **arg)
{
  if (!allocated) allocate();
  const int ntypes = atom->ntypes;
  if (narg != ntypes + 3)
    error->all(FLERR,
               "pair_coeff for mamba must be '* * model.mace.pt element1 ... elementN'");
  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "Pair style mamba requires pair_coeff * *");

  select_device();
  if (intraop_threads > 0) at::set_num_threads(intraop_threads);
  model_path = utils::get_potential_file_path(arg[2]);
  if (comm->me == 0) utils::logmesg(lmp, "MAMBA-ACE: loading {}\n", model_path);
  load_model(model_path);

  type_to_atomic_number.assign(ntypes + 1, 0);
  for (int type = 1; type <= ntypes; ++type) {
    const std::string name = arg[type + 2];
    const auto found = std::find(model_elements.begin(), model_elements.end(), name);
    if (found == model_elements.end())
      error->all(FLERR, "LAMMPS type {} element '{}' is absent from model metadata", type, name);
    const auto model_index = static_cast<size_t>(std::distance(model_elements.begin(), found));
    type_to_atomic_number[type] = model_atomic_numbers[model_index];
    if (comm->me == 0)
      utils::logmesg(lmp, "MAMBA-ACE: LAMMPS type {} -> {} (Z={})\n", type, name,
                     type_to_atomic_number[type]);
  }
  for (int i = 1; i <= ntypes; ++i)
    for (int j = i; j <= ntypes; ++j) setflag[i][j] = 1;

  if (comm->me == 0)
    utils::logmesg(lmp, "MAMBA-ACE: cutoff {:.8g} Angstrom, device {}, dtype {}\n", cutoff,
                   device.str(), model_dtype == torch::kFloat32 ? "float32" : "float64");
}

double PairMAMBA::init_one(int, int)
{
  return cutoff;
}

void PairMAMBA::check_deployment_contract() const
{
  if (!model_loaded) error->all(FLERR, "Pair style mamba coefficients are not set");
  if (force->newton_pair == 0) error->all(FLERR, "Pair style mamba requires newton pair on");
  if (atom->molecular != Atom::ATOMIC)
    for (int level = 1; level <= 3; ++level) {
      if (force->special_lj[level] == 0.0 && force->special_coul[level] == 0.0)
        error->all(FLERR,
                   "Pair style mamba requires every 1-2, 1-3, and 1-4 neighbor to remain "
                   "in the LAMMPS list; LJ and Coulomb special weights cannot both be zero");
    }
}

void PairMAMBA::init_style()
{
  check_deployment_contract();
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

PairMAMBA::Evaluation PairMAMBA::evaluate(torch::Tensor atomic_numbers,
                                          torch::Tensor positions, torch::Tensor edges,
                                          int nlocal, bool compute_energy, bool compute_virial)
{
  c10::DeviceGuard device_guard(device);
  torch::AutoGradMode enable_grad(true);
  positions = positions.detach().set_requires_grad(true);
  std::vector<torch::Tensor> gradient_inputs = {positions};
  torch::Tensor strain;
  torch::Tensor deformed_positions = positions;
  if (compute_virial) {
    strain = torch::zeros({3, 3}, positions.options().requires_grad(true));
    deformed_positions = torch::matmul(
        positions, torch::eye(3, positions.options()) + strain);
    gradient_inputs.push_back(strain);
  }

  torch::Tensor atomic_energy;
  try {
    atomic_energy = model.forward({atomic_numbers, deformed_positions, edges}).toTensor();
  } catch (const c10::Error &exception) {
    error->all(FLERR, "MAMBA-ACE model evaluation failed: {}", exception.what());
  }
  if (atomic_energy.dim() != 1 || atomic_energy.size(0) != positions.size(0))
    error->all(FLERR, "MAMBA-ACE model returned invalid atomic-energy shape");
  if (atomic_energy.scalar_type() != model_dtype || atomic_energy.device() != device)
    error->all(FLERR, "MAMBA-ACE model returned atomic energies with the wrong dtype or device");
  auto local_energy = atomic_energy.narrow(0, 0, nlocal).sum();
  // Preserve a differentiable zero for isolated configurations.
  local_energy = local_energy + positions.sum() * 0.0;
  if (compute_virial) local_energy = local_energy + strain.sum() * 0.0;

  std::vector<torch::Tensor> gradients;
  try {
    gradients = torch::autograd::grad({local_energy}, gradient_inputs, {}, false, false, true);
  } catch (const c10::Error &exception) {
    error->all(FLERR, "MAMBA-ACE automatic differentiation failed: {}", exception.what());
  }
  if (gradients.size() != gradient_inputs.size())
    error->all(FLERR, "MAMBA-ACE automatic differentiation returned the wrong gradient count");
  Evaluation result;
  result.atomic_energy = atomic_energy.detach();
  result.forces = gradients[0].defined() ? -gradients[0].detach() : torch::zeros_like(positions);
  if (compute_energy) result.total_energy = local_energy.detach().item<double>();
  if (compute_virial) {
    auto gradient = gradients[1].defined() ? gradients[1] : torch::zeros_like(strain);
    result.virial = (-0.5 * (gradient + gradient.transpose(0, 1))).detach();
  }
  if (check_finite) {
    bool finite = false;
    try {
      finite = torch::isfinite(result.atomic_energy.narrow(0, 0, nlocal)).all().item<bool>() &&
          torch::isfinite(result.forces).all().item<bool>();
      if (compute_energy) finite = finite && std::isfinite(result.total_energy);
      if (compute_virial) finite = finite && torch::isfinite(result.virial).all().item<bool>();
    } catch (const c10::Error &exception) {
      error->all(FLERR, "MAMBA-ACE finite-value validation failed: {}", exception.what());
    }
    if (!finite)
      error->all(FLERR, "MAMBA-ACE produced a nonfinite energy, force, or virial");
  }
  return result;
}

void PairMAMBA::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);
  if (vflag_atom || cvflag_atom)
    error->all(FLERR, "Pair style mamba does not define a per-atom virial");

  const int nlocal = atom->nlocal;
  const int nall = nlocal + atom->nghost;
  if (nlocal == 0) return;
  if (list->inum != nlocal)
    error->all(FLERR, "Pair style mamba requires all owned atoms in its full neighbor list");

  double **x = atom->x;
  int *type = atom->type;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  const double cutoff_sq = cutoff * cutoff;
  const float cutoff_float = static_cast<float>(cutoff);
  const float cutoff_sq_float = cutoff_float * cutoff_float;
  const double origin[3] = {
      0.5 * (domain->boxlo[0] + domain->boxhi[0]),
      0.5 * (domain->boxlo[1] + domain->boxhi[1]),
      0.5 * (domain->boxlo[2] + domain->boxhi[2])};

  std::vector<int64_t> counts(nlocal, 0);
#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int ii = 0; ii < nlocal; ++ii) {
    const int i = ilist[ii];
    int64_t count = 0;
    for (int jj = 0; jj < numneigh[i]; ++jj) {
      const int j = firstneigh[i][jj] & NEIGHMASK;
      bool inside;
      if (model_dtype == torch::kFloat32) {
        const float dx = static_cast<float>(x[j][0] - origin[0]) -
            static_cast<float>(x[i][0] - origin[0]);
        const float dy = static_cast<float>(x[j][1] - origin[1]) -
            static_cast<float>(x[i][1] - origin[1]);
        const float dz = static_cast<float>(x[j][2] - origin[2]) -
            static_cast<float>(x[i][2] - origin[2]);
        inside = dx * dx + dy * dy + dz * dz < cutoff_sq_float;
      } else {
        const double dx = x[j][0] - x[i][0];
        const double dy = x[j][1] - x[i][1];
        const double dz = x[j][2] - x[i][2];
        inside = dx * dx + dy * dy + dz * dz < cutoff_sq;
      }
      if (inside) ++count;
    }
    counts[ii] = count;
  }
  std::vector<int64_t> offsets(nlocal + 1, 0);
  for (int ii = 0; ii < nlocal; ++ii) offsets[ii + 1] = offsets[ii] + counts[ii];
  const int64_t nedges = offsets[nlocal];

  const bool pinned = device.is_cuda();
  auto cpu_float = torch::TensorOptions().dtype(model_dtype).device(torch::kCPU).pinned_memory(pinned);
  auto cpu_long = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU).pinned_memory(pinned);
  auto host_positions = torch::empty({nall, 3}, cpu_float);
  auto host_numbers = torch::empty({nall}, cpu_long);
  auto host_edges = torch::empty({2, nedges}, cpu_long);

  auto numbers = host_numbers.accessor<int64_t, 1>();
  auto edge = host_edges.accessor<int64_t, 2>();
  if (model_dtype == torch::kFloat32) {
    auto position = host_positions.accessor<float, 2>();
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int i = 0; i < nall; ++i) {
      numbers[i] = type_to_atomic_number[type[i]];
      for (int axis = 0; axis < 3; ++axis) position[i][axis] = static_cast<float>(x[i][axis] - origin[axis]);
    }
  } else {
    auto position = host_positions.accessor<double, 2>();
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int i = 0; i < nall; ++i) {
      numbers[i] = type_to_atomic_number[type[i]];
      for (int axis = 0; axis < 3; ++axis) position[i][axis] = x[i][axis] - origin[axis];
    }
  }
#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int ii = 0; ii < nlocal; ++ii) {
    const int i = ilist[ii];
    int64_t index = offsets[ii];
    for (int jj = 0; jj < numneigh[i]; ++jj) {
      const int j = firstneigh[i][jj] & NEIGHMASK;
      bool inside;
      if (model_dtype == torch::kFloat32) {
        const float dx = static_cast<float>(x[j][0] - origin[0]) -
            static_cast<float>(x[i][0] - origin[0]);
        const float dy = static_cast<float>(x[j][1] - origin[1]) -
            static_cast<float>(x[i][1] - origin[1]);
        const float dz = static_cast<float>(x[j][2] - origin[2]) -
            static_cast<float>(x[i][2] - origin[2]);
        inside = dx * dx + dy * dy + dz * dz < cutoff_sq_float;
      } else {
        const double dx = x[j][0] - x[i][0];
        const double dy = x[j][1] - x[i][1];
        const double dz = x[j][2] - x[i][2];
        inside = dx * dx + dy * dy + dz * dz < cutoff_sq;
      }
      if (!inside) continue;
      edge[0][index] = j;
      edge[1][index] = i;
      ++index;
    }
  }

  try {
    c10::DeviceGuard device_guard(device);
    auto positions = device.is_cpu() ? host_positions :
                                       host_positions.to(device, model_dtype, true, true);
    auto atomic_numbers = device.is_cpu() ? host_numbers :
                                            host_numbers.to(device, torch::kInt64, true, true);
    auto edges = device.is_cpu() ? host_edges :
                                   host_edges.to(device, torch::kInt64, true, true);
    auto result = evaluate(atomic_numbers, positions, edges, nlocal, eflag_global != 0,
                           vflag_global != 0);
    auto forces = result.forces.to(torch::kCPU).contiguous();
    torch::Tensor atomic_energy;
    if (eflag_atom) atomic_energy = result.atomic_energy.to(torch::kCPU).contiguous();

    double **f = atom->f;
    if (model_dtype == torch::kFloat32) {
      auto force_values = forces.accessor<float, 2>();
      for (int i = 0; i < nall; ++i)
        for (int axis = 0; axis < 3; ++axis) f[i][axis] += force_values[i][axis];
      if (eflag_atom) {
        auto energy_values = atomic_energy.accessor<float, 1>();
        for (int i = 0; i < nlocal; ++i) eatom[i] += energy_values[i];
      }
    } else {
      auto force_values = forces.accessor<double, 2>();
      for (int i = 0; i < nall; ++i)
        for (int axis = 0; axis < 3; ++axis) f[i][axis] += force_values[i][axis];
      if (eflag_atom) {
        auto energy_values = atomic_energy.accessor<double, 1>();
        for (int i = 0; i < nlocal; ++i) eatom[i] += energy_values[i];
      }
    }
    if (eflag_global) eng_vdwl = result.total_energy;
    if (vflag_global) {
      auto virial_cpu = result.virial.to(torch::kCPU, torch::kFloat64).contiguous();
      auto value = virial_cpu.accessor<double, 2>();
      virial[0] = value[0][0];
      virial[1] = value[1][1];
      virial[2] = value[2][2];
      virial[3] = value[0][1];
      virial[4] = value[0][2];
      virial[5] = value[1][2];
    }
  } catch (const c10::Error &exception) {
    error->all(FLERR, "MAMBA-ACE LibTorch execution failed: {}", exception.what());
  }

  if (debug && comm->me == 0)
    utils::logmesg(lmp, "MAMBA-ACE: nlocal={}, nall={}, directed_edges={}\n", nlocal, nall,
                   nedges);
}
