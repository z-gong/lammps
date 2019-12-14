/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(temp/drude,ComputeTempDrude)

#else

#ifndef LMP_COMPUTE_TEMP_DRUDE_H
#define LMP_COMPUTE_TEMP_DRUDE_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeTempDrude : public Compute {
 public:
  ComputeTempDrude(class LAMMPS *, int, char **);
  ~ComputeTempDrude();
  void init();
  void setup();
  void compute_vector();
  double compute_scalar();
  int modify_param(int, char **);

 private:
  int fix_dof;
  class FixDrude * fix_drude;
  char *id_temp;
  class Compute *temperature;
  bigint dof_core, dof_drude;
  double kineng_core, kineng_drude;
  double temp_core, temp_drude;

  void dof_compute();

  /* ------------------------------------
   * copy from fix_tgnh.cpp
   * ------------------------------------ */
  int n_mol;                            // number of molecules in the system
  double *mass_mol;
  double dof_mol, dof_int;   // DOFs of different modes in the fix group
  double **v_mol, **v_mol_tmp;

  void compute_temp_mol_int_drude();
  double t_mol, t_int, t_drude;
  double ke2mol, ke2int, ke2drude;
  /* ------------------------------------
   * end of copy
   * ------------------------------------ */

};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

*/
