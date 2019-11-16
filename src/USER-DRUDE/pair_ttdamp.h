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

#ifdef PAIR_CLASS

PairStyle(ttdamp,PairTTDamp)

#else

#ifndef LMP_PAIR_TTDAMP_H
#define LMP_PAIR_TTDAMP_H

#include "pair.h"
#include <vector>

namespace LAMMPS_NS {

class PairTTDamp: public Pair {
 public:
  PairTTDamp(class LAMMPS *);
  virtual ~PairTTDamp();
  virtual void compute(int, int);
  virtual void settings(int, char **);
  void coeff(int, char **);
  void init_style();
  double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  virtual void write_restart_settings(FILE *);
  virtual void read_restart_settings(FILE *);
  virtual double single(int, int, int, int, double, double, double, double &);
  void *extract(const char *, int &);

 protected:
  int n_global;
  std::vector<int> factorial;
//  int factorial[9] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320};
  double cut_global;
  double **cut,**scale;
  double **b,**c;
  int **n;
  class FixDrude * fix_drude;

  virtual void allocate();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair style thole requires atom attribute q

The atom style defined does not have these attributes.

*/
