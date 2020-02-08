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

#include "fix_hardwall_drude.h"
#include <mpi.h>
#include <cstring>
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
#include "neighbor.h"
#include "fix_drude.h"

using namespace LAMMPS_NS;
using namespace FixConst;

enum {
  NONE, CONSTANT, EQUAL, ATOM
};

/* ---------------------------------------------------------------------- */

FixHardwallDrude::FixHardwallDrude(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg) {
  if (narg != 5) error->all(FLERR, "Illegal fix hardwall/drude command");

  dynamic_group_allow = 1;
  scalar_flag = 1;
  extscalar = 1;

  limit = force->numeric(FLERR, arg[3]);
  t_drude = force->numeric(FLERR, arg[4]);

  n_bad_bond = 0;
  fix_drude = NULL;
}

/* ---------------------------------------------------------------------- */

FixHardwallDrude::~FixHardwallDrude() {
}

/* ---------------------------------------------------------------------- */

int FixHardwallDrude::setmask() {
  int mask = 0;
  mask |= POST_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixHardwallDrude::init() {
  int ifix;
  for (ifix = 0; ifix < modify->nfix; ifix++)
    if (strcmp(modify->fix[ifix]->style, "drude") == 0) break;
  if (ifix == modify->nfix) error->all(FLERR, "fix hardwall/drude requires fix drude");
  fix_drude = (FixDrude *) modify->fix[ifix];
}

/* ---------------------------------------------------------------------- */

void FixHardwallDrude::post_integrate() {
  double **x = atom->x;
  double **v = atom->v;
  int *type = atom->type;
  int *mask = atom->mask;
  double *mass = atom->mass;

  int *drudetype = fix_drude->drudetype;
  tagint *drudeid = fix_drude->drudeid;

  int i, j, itype, ci, di;
  double mass_com, mass_core, mass_drude;
  double bond[3], bondDir[3];
  double vcore_bond[3], vdrude_bond[3], vcore_normal[3], vdrude_normal[3];

  n_bad_bond = 0;
  for (i = 0; i < atom->nlocal; i++) {
    itype = drudetype[type[i]];
    if (mask[i] & groupbit && itype != NOPOL_TYPE) {
      j = domain->closest_image(i, atom->map(drudeid[i]));
      if (itype == DRUDE_TYPE && j < atom->nlocal) continue;
      if (itype == CORE_TYPE) {
        ci = i;
        di = j;
      } else {
        ci = j;
        di = i;
      }
      mass_core = mass[type[ci]];
      mass_drude = mass[type[di]];
      mass_com = mass_core + mass_drude;

      for (int k = 0; k < 3; k++) {
        bond[k] = x[di][k] - x[ci][k];
      }
      double r = sqrt(dot(bond, bond));
      if (r <= limit) continue;
      if (r >= limit * 2) {
        char str[1024];
        sprintf(str, "Distance of Drude pair exceeds twice the hardwall limit: %d %d %.4f",
                atom->tag[ci], atom->tag[di], r);
        error->all(FLERR, str);
      }

      n_bad_bond++;

      multiply(bond, 1.0 / r, bondDir);
      double vcore_bond_scalar = dot(v[ci], bondDir);
      double vdrude_bond_scalar = dot(v[di], bondDir);
      double vcom_bond_scalar = (mass_core * vcore_bond_scalar + mass_drude * vdrude_bond_scalar) / mass_com;

      for (int k = 0; k < 3; k++) {
        vcore_bond[k] = bondDir[k] * vcore_bond_scalar;
        vdrude_bond[k] = bondDir[k] * vdrude_bond_scalar;
        vcore_normal[k] = v[ci][k] - vcore_bond[k];
        vdrude_normal[k] = v[di][k] - vdrude_bond[k];
      }

      // copy from OpenMM implementation
      // it's different from the original reference
      // do not understand why
      double deltaR = r - limit;
      double dt = update->dt;
      if (vdrude_bond_scalar != vcore_bond_scalar)
        dt = std::min(dt, deltaR / std::fabs(vdrude_bond_scalar - vcore_bond_scalar));
      double scale = sqrt(force->boltz * t_drude / mass_drude / force->mvv2e);

      vcore_bond_scalar -= vcom_bond_scalar;
      vdrude_bond_scalar -= vcom_bond_scalar;

      vcore_bond_scalar = -vcore_bond_scalar * mass_drude / std::fabs(vcore_bond_scalar) / mass_com * scale;
      vdrude_bond_scalar = -vdrude_bond_scalar * mass_core / std::fabs(vdrude_bond_scalar) / mass_com * scale;

      double dr_core_scalar = deltaR * mass_drude / mass_com + dt * vcore_bond_scalar;
      double dr_drude_scalar = -deltaR * mass_core / mass_com + dt * vdrude_bond_scalar;
      for (int k = 0; k < 3; k++) {
        x[ci][k] += bondDir[k] * dr_core_scalar;
        x[di][k] += bondDir[k] * dr_drude_scalar;
      }

      vcore_bond_scalar += vcom_bond_scalar;
      vdrude_bond_scalar += vcom_bond_scalar;
      for (int k = 0; k < 3; k++) {
        v[ci][k] = bondDir[k] * vcore_bond_scalar + vcore_normal[k];
        v[di][k] = bondDir[k] * vdrude_bond_scalar + vdrude_normal[k];
      }
      // since hard wall is a remedy for rare event, the contribution to Virial is ignored
    }
  }
}

double FixHardwallDrude::compute_scalar()
{
  int scalar;
  MPI_Allreduce(&n_bad_bond,&scalar,1,MPI_INT,MPI_SUM,world);
  return scalar;
}
