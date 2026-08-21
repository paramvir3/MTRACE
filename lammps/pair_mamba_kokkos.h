/* -*- c++ -*- ----------------------------------------------------------
   Kokkos device interface for MAMBA-ACE.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(mamba/kk, PairMAMBAKokkos<LMPDeviceType>)

#else

#ifndef LMP_PAIR_MAMBA_KOKKOS_H
#define LMP_PAIR_MAMBA_KOKKOS_H

#include "pair_mamba.h"
#include "pair_kokkos.h"

namespace LAMMPS_NS {

template <class DeviceType> class PairMAMBAKokkos : public PairMAMBA {
 public:
  using device_type = DeviceType;
  using AT = ArrayTypes<DeviceType>;
  enum { EnabledNeighFlags = FULL | HALF | HALFTHREAD };
  enum { COUL_FLAG = 0 };

  PairMAMBAKokkos(class LAMMPS *);
  ~PairMAMBAKokkos() override;
  void compute(int, int) override;
  void coeff(int, char **) override;
  void init_style() override;

 private:
  using Int64View = Kokkos::View<int64_t *, Kokkos::LayoutRight, DeviceType>;
  using Int64View2D = Kokkos::View<int64_t **, Kokkos::LayoutRight, DeviceType>;
  using IntView = Kokkos::View<int *, Kokkos::LayoutRight, DeviceType>;
  using FloatView2D = Kokkos::View<float **, Kokkos::LayoutRight, DeviceType>;
  using DoubleView2D = Kokkos::View<double **, Kokkos::LayoutRight, DeviceType>;

  class AtomKokkos *atomKK = nullptr;
  typename AT::t_kkfloat_1d_3_lr_randomread x;
  typename AT::t_kkacc_1d_3 f;
  typename AT::t_int_1d_randomread type;
  typename AT::t_neighbors_2d neighbors;
  typename AT::t_int_1d_randomread ilist;
  typename AT::t_int_1d_randomread numneigh;
  typename AT::t_kkacc_1d d_eatom;
  DAT::ttransform_kkacc_1d k_eatom;

  Int64View type_to_z;
  Int64View atomic_numbers;
  Int64View2D edges;
  IntView edge_counts;
  Int64View edge_offsets;
  FloatView2D positions_float;
  DoubleView2D positions_double;
  int neighflag = 0;
};

}    // namespace LAMMPS_NS

#endif
#endif
