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

PairStyle(coul/diel/gz,PairCoulDielGZ)

#else

#ifndef LMP_PAIR_COUL_DIEL_GZ_H
#define LMP_PAIR_COUL_DIEL_GZ_H

#include "pair.h"

namespace LAMMPS_NS {

class PairCoulDielGZ : public Pair {
 public:
  PairCoulDielGZ(class LAMMPS *);
  virtual ~PairCoulDielGZ();

  virtual void compute(int, int);

  virtual void settings(int, char **);
  virtual void coeff(int, char **);
  virtual void init_style();
  virtual double init_one(int, int);
  virtual void write_restart(FILE *);
  virtual void read_restart(FILE *);
  virtual void write_restart_settings(FILE *);
  virtual void read_restart_settings(FILE *);

  virtual double single(int, int, int, int, double, double, double, double &);

 protected:
  double cut_global;
  double **cut;
  double **sigmae, **rme, **offset;
  double **eps_0, **eps_s, **eps_a, **eps_b;

  void allocate();
};

}

#endif
#endif
