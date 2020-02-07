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

FixStyle(drude/hardwall,FixDrudeHardwall)

#else

#ifndef LMP_FIX_DRUDE_HARDWALL_H
#define LMP_FIX_DRUDE_HARDWALL_H

#include "fix.h"

namespace LAMMPS_NS {

class FixDrudeHardwall : public Fix {
 public:
  FixDrudeHardwall(class LAMMPS *, int, char **);
  virtual ~FixDrudeHardwall();
  int setmask();
  void init();
  void post_integrate();
  double compute_scalar();

  double limit, t_drude;
  int n_bad_bond;

  class FixDrude * fix_drude;

  static inline double dot(const double *a, const double *b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }

  static inline void add(const double *a, const double *b, double *result) {
    for (int k = 0; k < 3; k++)
      result[k] = a[k] + b[k];
  }

  static inline void subtract(const double *a, const double *b, double *result) {
    for (int k = 0; k < 3; k++)
      result[k] = a[k] - b[k];
  }

  static inline void multiply(const double *v, const double times, double *result) {
    for (int k = 0; k < 3; k++)
      result[k] = v[k] * times;
  }

};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Region ID for fix setforce does not exist

Self-explanatory.

E: Variable name for fix setforce does not exist

Self-explanatory.

E: Variable for fix setforce is invalid style

Only equal-style variables can be used.

E: Cannot use non-zero forces in an energy minimization

Fix setforce cannot be used in this manner.  Use fix addforce
instead.

*/
