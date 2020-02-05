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
   Contributing author: Zheng GONG (ENS de Lyon)
------------------------------------------------------------------------- */

#include "fix_imagecharge.h"
#include <cmath>
#include <cstring>
#include "atom.h"
#include "update.h"
#include "force.h"
#include "group.h"
#include "error.h"
#include "memory.h"
#include "comm.h"
#include <vector>
#include "kspace.h"
#include "domain.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixImageCharge::FixImageCharge(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg) {
  scalar_flag = 0;
  vector_flag = 0;
  size_vector = 4;
  global_freq = 1;
  extscalar = 1;
  extvector = 1;
  dynamic_group_allow = 0;

  if (narg != 5) error->all(FLERR, "Illegal fix imagecharge command");

  igroup2 = group->find(arg[3]);
  if (igroup2 == -1)
    error->all(FLERR, "Fix imagecharge active group ID does not exist");
  if (group->count(igroup) != group->count(igroup2))
    error->all(FLERR, "Fix imagecharge number of image atoms does not equal to electrolyte atoms");
  group2bit = group->bitmask[igroup2];

  z0 = force->numeric(FLERR, arg[4]);

  memory->create(img_parent, atom->natoms + 1, "fix_image_charge::img_parent");
  memory->create(xyz, atom->natoms + 1, 3, "fix_image_charge::xyz");
  memory->create(xyz_local, atom->natoms + 1, 3, "fix_image_charge::xyz_local");
  memory->create(xyz_img_tmp, 3, "fix_image_charge::xyz_img_tmp");
  memory->create(charge, atom->natoms + 1, "fix_image_charge::charge");
  memory->create(charge_local, atom->natoms + 1, "fix_image_charge::charge_local");
}

/* ---------------------------------------------------------------------- */

FixImageCharge::~FixImageCharge() {
  memory->destroy(img_parent);
  memory->destroy(xyz);
  memory->destroy(xyz_local);
  memory->destroy(xyz_img_tmp);
  memory->destroy(charge);
  memory->destroy(charge_local);
}

/* ---------------------------------------------------------------------- */

int FixImageCharge::setmask() {
  int mask = 0;
  mask |= PRE_EXCHANGE;
  mask |= PRE_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixImageCharge::init() {
  build_img_parents();
  // assume fixed charge model. Only assign image charges once
  assign_img_charges();
  update_img_positions();
}

/* ---------------------------------------------------------------------- */

void FixImageCharge::pre_exchange(int /*vflag*/) {
  update_img_positions();
}

/* ---------------------------------------------------------------------- */

void FixImageCharge::pre_force(int /*vflag*/) {
  update_img_positions();
}

/* ---------------------------------------------------------------------- */
void FixImageCharge::build_img_parents() {
  int *flag_parents = new int[atom->natoms + 1];
  int *flag_imgs = new int[atom->natoms + 1];
  int *flag_parents_local = new int[atom->natoms + 1];
  int *flag_imgs_local = new int[atom->natoms + 1];
  std::vector<int> tag_parents, tag_imgs;
  int *mask = atom->mask;

  for (int i = 1; i < atom->natoms + 1; i++) {
    flag_parents_local[i] = 0;
    flag_imgs_local[i] = 0;
  }
  for (int ii = 0; ii < atom->nlocal; ii++) {
    if (mask[ii] & groupbit) {
      flag_parents_local[atom->tag[ii]] = 1;
    }
    if (mask[ii] & group2bit) {
      flag_imgs_local[atom->tag[ii]] = 1;
    }
  }
  MPI_Allreduce(flag_parents_local, flag_parents, atom->natoms + 1, MPI_INT, MPI_SUM, world);
  MPI_Allreduce(flag_imgs_local, flag_imgs, atom->natoms + 1, MPI_INT, MPI_SUM, world);

  for (int i = 1; i < atom->natoms + 1; i++) {
    img_parent[i] = -1;
    if (flag_parents[i] == 1) tag_parents.push_back(i);
    if (flag_imgs[i] == 1) tag_imgs.push_back(i);
  }

  for (int i = 0; i < tag_parents.size(); i++) {
    img_parent[tag_imgs[i]] = tag_parents[i];
  }
}

/* ---------------------------------------------------------------------- */

void FixImageCharge::assign_img_charges() {
  int tag, tag_parent;
  int *mask = atom->mask;
  double *q = atom->q;

  // initialize the local charge array
  for (int i = 1; i < atom->natoms + 1; i++) {
    charge_local[i] = 0.0;
  }

  // store the charge of parent atoms
  for (int ii = 0; ii < atom->nlocal; ii++) {
    if (mask[ii] & groupbit) {
      tag = atom->tag[ii];
      charge_local[tag] = q[ii];
    }
  }
  MPI_Allreduce(charge_local, charge, atom->natoms + 1, MPI_DOUBLE, MPI_SUM, world);

  // update the charge of image particles based on their parents
  for (int ii = 0; ii < atom->nlocal; ii++) {
    tag = atom->tag[ii];
    tag_parent = img_parent[tag];
    if (tag_parent != -1) {
      q[ii] = -charge[tag_parent];
    }
  }

  // reset KSpace charges
  if (force->kspace) force->kspace->qsum_qsq();
}

/* ---------------------------------------------------------------------- */

void FixImageCharge::update_img_positions() {
  int tag, tag_parent;
  int *mask = atom->mask;
  double **x = atom->x;

  // initialize the local xyz array.
  // actually can be put in pre_neighbour(). Put here for clarity
  for (int i = 1; i < atom->natoms + 1; i++) {
    xyz_local[i][0] = 0.0;
    xyz_local[i][1] = 0.0;
    xyz_local[i][2] = 0.0;
  }

  // store the xyz of parent atoms
  for (int ii = 0; ii < atom->nlocal; ii++) {
    if (mask[ii] & groupbit) {
      tag = atom->tag[ii];
      xyz_local[tag][0] = x[ii][0];
      xyz_local[tag][1] = x[ii][1];
      xyz_local[tag][2] = x[ii][2];
    }
  }
  MPI_Allreduce(*xyz_local, *xyz, (atom->natoms + 1) * 3, MPI_DOUBLE, MPI_SUM, world);

  // update the xyz of image particles
  for (int ii = 0; ii < atom->nlocal; ii++) {
    tag = atom->tag[ii];
    tag_parent = img_parent[tag];
    if (tag_parent != -1) {
      xyz_img_tmp[0] = x[ii][0];
      xyz_img_tmp[1] = x[ii][1];
      xyz_img_tmp[2] = x[ii][2];
      x[ii][0] = xyz[tag_parent][0];
      x[ii][1] = xyz[tag_parent][1];
      x[ii][2] = 2 * z0 - xyz[tag_parent][2];
      // make sure the new coordinate of images is in the correct periodic box
       domain->remap_near(x[ii], xyz_img_tmp);
//      if (abs(x[ii][0] - xyz_tmp[0]) > 0.01 or abs(x[ii][1] - xyz_tmp[1]) > 0.01 or abs(x[ii][2] - xyz_tmp[2]) > 0.01)
//        printf("old and new xyz: %d %d, %f %f %f, %f %f %f\n", tag, tag_parent,
//               xyz_tmp[0], xyz_tmp[1], xyz_tmp[2], x[ii][0], x[ii][1], x[ii][2]);
    }
  }
}
