/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Zheng Gong
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(inter,ComputeInter)

#else

#ifndef LMP_COMPUTE_INTER_H
#define LMP_COMPUTE_INTER_H

#include "compute.h"

namespace LAMMPS_NS {

  class ComputeInter : public Compute {
    public:
      ComputeInter(class LAMMPS *, int , char **);
      ~ComputeInter();

      void init();
      void compute_vector();

      int pack_comm(int, int *, double *, int, int *);
      void unpack_comm(int, int, double *);
      double memory_usage();

    protected:
      int me;

      double eng_pair_intra;   // accumulated intramolecular energies
      bool ignore_tail; // ignore tail correction

      void compute_intra();
      double coul_kspace_correct(int i, int j, int itype, int jtype, double rsq, double factor_coul);

      /*enum SINGLE_MODE {NORMAL, TIP4P};*/
      /*SINGLE_MODE mode;*/
  };
}

#endif
#endif
