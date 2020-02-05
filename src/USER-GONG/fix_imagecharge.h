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

FixStyle(imagecharge,FixImageCharge)

#else

#ifndef LMP_FIX_IMAGE_CHARGE_H
#define LMP_FIX_IMAGE_CHARGE_H

#include "fix.h"

namespace LAMMPS_NS {

class FixImageCharge : public Fix {
 public:
  FixImageCharge(class LAMMPS *, int, char **);
  ~FixImageCharge();
  int setmask();
  void init();
  void pre_exchange(int);
  void pre_force(int);

 private:
  int igroup2,group2bit;

  double z0;
  tagint * img_parent;
  double ** xyz;
  double ** xyz_local;
  double * xyz_img_tmp;
  double * charge;
  double * charge_local;
  void build_img_parents();
  void assign_img_charges();
  void update_img_positions();
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
