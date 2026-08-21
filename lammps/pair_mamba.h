/* -*- c++ -*- ----------------------------------------------------------
   MAMBA-ACE pair style for LAMMPS.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(mamba, PairMAMBA)

#else

#ifndef LMP_PAIR_MAMBA_H
#define LMP_PAIR_MAMBA_H

#include "pair.h"

#include <torch/script.h>
#include <torch/torch.h>

#include <string>
#include <vector>

namespace LAMMPS_NS {

class PairMAMBA : public Pair {
 public:
  PairMAMBA(class LAMMPS *);
  ~PairMAMBA() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  double init_one(int, int) override;
  void init_style() override;

 protected:
  struct Evaluation {
    torch::Tensor atomic_energy;
    torch::Tensor forces;
    torch::Tensor virial;
    double total_energy = 0.0;
  };

  void allocate();
  void select_device();
  void load_model(const std::string &);
  Evaluation evaluate(torch::Tensor, torch::Tensor, torch::Tensor, int, bool, bool);
  void check_deployment_contract() const;

  torch::jit::Module model;
  torch::Device device = torch::Device(torch::kCPU);
  torch::ScalarType model_dtype = torch::kFloat32;
  std::string requested_device = "auto";
  std::string model_path;
  std::vector<int64_t> type_to_atomic_number;
  std::vector<std::string> model_elements;
  std::vector<int64_t> model_atomic_numbers;
  double cutoff = 0.0;
  bool model_loaded = false;
  bool debug = false;
  bool check_finite = true;
  int intraop_threads = 0;
};

}    // namespace LAMMPS_NS

#endif
#endif
