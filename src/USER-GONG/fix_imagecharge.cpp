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
   Contributing author: Paul Crozier (SNL)
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
#include "stdlib.h"

using namespace LAMMPS_NS;
using namespace FixConst;

#define SMALL 1.0e-10

enum {
    TETHER, COUPLE
};

/* ---------------------------------------------------------------------- */

FixImageCharge::FixImageCharge(LAMMPS *lmp, int narg, char **arg) :
        Fix(lmp, narg, arg),
        group2(NULL) {
  scalar_flag = 0;
  vector_flag = 0;
  size_vector = 4;
  global_freq = 1;
  extscalar = 1;
  extvector = 1;
  dynamic_group_allow = 0;

  if (narg != 5) error->all(FLERR, "Illegal fix imgq command");

  int n = strlen(arg[3]) + 1;
  group2 = new char[n];
  strcpy(group2, arg[3]);
  igroup2 = group->find(arg[3]);
  if (igroup2 == -1)
    error->all(FLERR, "Fix imgq active group ID does not exist");
  if (group->count(igroup) != group->count(igroup2))
    error->all(FLERR, "Fix imgq number of atoms in image charge group does not equal to active group");
  group2bit = group->bitmask[igroup2];

  z0 = force->numeric(FLERR, arg[4]);

  memory->create(img_parent, atom->natoms + 1, "fix_imgq::img_parent");
  memory->create(xyz, (atom->natoms + 1) * 3, "fix_imgq::xyz");
  memory->create(xyz_local, (atom->natoms + 1) * 3, "fix_imgq::xyz_local");
}

/* ---------------------------------------------------------------------- */

FixImageCharge::~FixImageCharge() {
  delete[] group2;
  memory->destroy(img_parent);
  memory->destroy(xyz);
  memory->destroy(xyz_local);
}

/* ---------------------------------------------------------------------- */

int FixImageCharge::setmask() {
  int mask = 0;
  mask |= PRE_FORCE;
  mask |= MIN_PRE_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixImageCharge::init() {
  build_img_parents();
}

/* ---------------------------------------------------------------------- */

void FixImageCharge::pre_force(int /*vflag*/) {
  update_img_positions();
}

/* ---------------------------------------------------------------------- */

void FixImageCharge::min_pre_force(int /*vflag*/) {
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
  // print parent atoms and image charges
  if (comm->me==0) {
    for (int tag: tag_parents) {
      printf("%d ", tag);
    }
    printf("\n");
    for (int tag: tag_imgs) {
      printf("%d ", tag);
    }
    printf("\n");
  }

  for (int i = 0; i < tag_parents.size(); i++) {
    img_parent[tag_imgs[i]] = tag_parents[i];
  }

  // set xyz array to zero. not really necessary
  for (int i = 1; i < atom->natoms + 1; i++) {
    xyz_local[3 * i + 0] = 0.0;
    xyz_local[3 * i + 1] = 0.0;
    xyz_local[3 * i + 2] = 0.0;
  }
}

/* ---------------------------------------------------------------------- */

void FixImageCharge::update_img_positions() {
  int tag, tag_parent;
  int *mask = atom->mask;
  double **x = atom->x;

  // store the xyz of parent atoms
  for (int ii = 0; ii < atom->nlocal; ii++) {
    tag = atom->tag[ii];
    if (mask[ii] & groupbit) {
      xyz_local[3 * tag + 0] = x[ii][0];
      xyz_local[3 * tag + 1] = x[ii][1];
      xyz_local[3 * tag + 2] = x[ii][2];
//      printf("%d %d %d, %f %f %f\n", comm->me, i, tag, x_tmp[tag], y_tmp[tag], z_tmp[tag]);
    }
  }
  MPI_Allreduce(xyz_local, xyz, (atom->natoms + 1) * 3, MPI_DOUBLE, MPI_SUM, world);

//  if (comm->me == 0){
//    for (int i = 1; i < atom->natoms + 1; i++) {
//      printf("%d %f %f %f\n", i, x_active[i], y_active[i], z_active[i]);
//    }
//  }

  // update the xyz of image charges
  for (int i = 0; i < atom->nlocal + atom->nghost; i++) {
    tag = atom->tag[i];
    tag_parent = img_parent[tag];
    if (tag_parent != -1) {
//      printf("%d %d %d, %f %f %f\n",i, tagimg, tag, x_active[tag], y_active[tag], z_active[tag]);
      x[i][0] = xyz[3 * tag_parent];
      x[i][1] = xyz[3 * tag_parent + 1];
      x[i][2] = 2 * z0 - xyz[3 * tag_parent + 2];
    }
  }
}
