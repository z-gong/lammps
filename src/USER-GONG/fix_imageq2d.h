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

#ifdef FIX_CLASS

FixStyle(imageq2d,FixImageQ2D)

#else

#ifndef LMP_FIX_IMAGE_Q2D_H
#define LMP_FIX_IMAGE_Q2D_H

#include "fix.h"

namespace LAMMPS_NS {

class FixImageQ2D : public Fix {
 public:
  FixImageQ2D(class LAMMPS *, int, char **);
  ~FixImageQ2D();
  int setmask();
  void init();
  void pre_exchange(int);
  void pre_force(int);

 private:
  int igroup2,group2bit;
  int igroup_cathode, groupbit_cathode;
  int igroup_anode, groupbit_anode;


  double V, z_cathode, z_anode, z_span; // voltage drop, position of cathode and anode
  double q_voltage; // charge because of the applied voltage drop
  int n_cathode, n_anode; // number of atoms in cathode and anode
  tagint * img_parent;
  double ** xyz;
  double ** xyz_local;
  double * xyz_img_tmp;
  double * charge;
  double * charge_local;
  void build_img_parents();
  void assign_img_charges();
  void update_img_positions();
  void update_electrode_charges();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: R0 < 0 for fix spring command

Equilibrium spring length is invalid.

E: Fix spring couple group ID does not exist

Self-explanatory.

E: Two groups cannot be the same in fix spring couple

Self-explanatory.

*/
