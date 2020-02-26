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

#include <string.h>
#include "fix_vis_pp.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "domain.h"
#include "region.h"
#include "respa.h"
#include "input.h"
#include "variable.h"
#include "memory.h"
#include "error.h"
#include "force.h"
#include "math_const.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixVisPP::FixVisPP(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg) {
  if (narg < 4) error->all(FLERR, "Illegal fix vis/pp command");
  acceleration = force->numeric(FLERR, arg[3]);
}

/* ---------------------------------------------------------------------- */

FixVisPP::~FixVisPP() {
}

/* ---------------------------------------------------------------------- */

int FixVisPP::setmask() {
  int mask = 0;
  mask |= POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixVisPP::init() {
}

/* ---------------------------------------------------------------------- */

void FixVisPP::setup(int vflag) {
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixVisPP::post_force(int vflag) {
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double massone, force_x, acc_x;
  double zlo = domain->boxlo[2];
  double zhi = domain->boxhi[2];

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      if (rmass) massone = rmass[i];
      else massone = mass[type[i]];

      acc_x = acceleration * cos(MathConst::MY_2PI * (x[i][2] - zlo) / (zhi - zlo));
      force_x = acc_x * massone * force->mvv2e; // unit from (g/mol)*(A/fs^2) to kcal/(mol.A)
//      printf("%f %f %f %f\n", mass[type[i]], x[i][2], acc_x, force_x);

      f[i][0] += force_x;
    }
}
