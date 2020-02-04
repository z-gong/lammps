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

#include "fix_imageq2d.h"
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

FixImageQ2D::FixImageQ2D(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg) {
  scalar_flag = 0;
  vector_flag = 0;
  size_vector = 4;
  global_freq = 1;
  extscalar = 1;
  extvector = 1;
  dynamic_group_allow = 0;

  if (narg != 9) error->all(FLERR, "Illegal fix imgq2d command");

  igroup2 = group->find(arg[3]);
  if (igroup2 == -1)
    error->all(FLERR, "Fix imgq2d image group ID does not exist");
  if (group->count(igroup) * 2 != group->count(igroup2))
    error->all(FLERR, "Fix imgq2d number of atoms in image group does not equal to two times of active group");
  group2bit = group->bitmask[igroup2];

  V = force->numeric(FLERR, arg[4]);

  igroup_cathode = group->find(arg[5]);
  if (igroup_cathode == -1)
    error->all(FLERR, "Fix imgq2d cathode group ID does not exist");
  groupbit_cathode = group->bitmask[igroup_cathode];

  z_cathode = force->numeric(FLERR, arg[6]);

  igroup_anode = group->find(arg[7]);
  if (igroup_anode == -1)
    error->all(FLERR, "Fix imgq2d anode group ID does not exist");
  groupbit_anode= group->bitmask[igroup_anode];

  z_anode = force->numeric(FLERR, arg[8]);
  z_span = z_anode - z_cathode;
  q_voltage = V * domain->xprd * domain->yprd / z_span * 8.85419 / 1.60218 * 0.001;
  n_cathode = group->count(igroup_cathode);
  n_anode = group->count(igroup_anode);

  if (comm->me == 0) {
    if (screen) {
      fprintf(screen, "Constant potential image charge method\n");
      fprintf(screen, "  charge %.1f spread on %d cathode atoms and %d anode atoms\n",
              q_voltage, n_cathode, n_anode);
    }
    if (logfile) {
      fprintf(logfile, "Constant potential image charge method\n");
      fprintf(logfile, "  charge %.1f spread on %d cathode atoms and %d anode atoms\n",
              q_voltage, n_cathode, n_anode);
    }
  }

  memory->create(img_parent, atom->natoms + 1, "fix_image_charge::img_parent");
  memory->create(xyz, atom->natoms + 1, 3, "fix_image_charge::xyz");
  memory->create(xyz_local, atom->natoms + 1, 3, "fix_image_charge::xyz_local");
  memory->create(xyz_img_tmp, 3, "fix_image_charge::xyz_img_tmp");
  memory->create(charge, atom->natoms + 1, "fix_image_charge::charge");
  memory->create(charge_local, atom->natoms + 1, "fix_image_charge::charge_local");
}

/* ---------------------------------------------------------------------- */

FixImageQ2D::~FixImageQ2D() {
  memory->destroy(img_parent);
  memory->destroy(xyz);
  memory->destroy(xyz_local);
  memory->destroy(xyz_img_tmp);
  memory->destroy(charge);
  memory->destroy(charge_local);
}

/* ---------------------------------------------------------------------- */

int FixImageQ2D::setmask() {
  int mask = 0;
  mask |= PRE_EXCHANGE;
  mask |= PRE_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixImageQ2D::init() {
  build_img_parents();
  // assume fixed charge model for electrolytes. Only assign image charges once
  assign_img_charges();
  update_img_positions();
  update_electrode_charges();
}

/* ---------------------------------------------------------------------- */

void FixImageQ2D::pre_exchange(int /*vflag*/) {
  update_img_positions();
  update_electrode_charges();
}

/* ---------------------------------------------------------------------- */

void FixImageQ2D::pre_force(int /*vflag*/) {
  update_img_positions();
  update_electrode_charges();
}

/* ---------------------------------------------------------------------- */
void FixImageQ2D::build_img_parents() {
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
    img_parent[i] = 0; // set the default of image_parent to 0. since the tag of atoms starts from 1. negative value is used for anode iamges
    if (flag_parents[i] == 1) tag_parents.push_back(i);
    if (flag_imgs[i] == 1) tag_imgs.push_back(i);
  }
  // print parent atoms and image particles
//  if (comm->me == 0) {
//    printf("Pairs of parent and image atoms\n  ");
//    for (int i = 0; i < tag_parents.size(); i++) {
//      printf("%d %d; ", tag_parents[i], tag_imgs[i]);
//    }
//    printf("\n");
//  }

  for (int i = 0; i < tag_parents.size(); i++) {
    img_parent[tag_imgs[i]] = tag_parents[i]; // images on cathode
    img_parent[tag_imgs[i] + tag_parents.size()] = -tag_parents[i]; // images on anode. set the tag of parent to negative so can be distinguished
  }
}

/* ---------------------------------------------------------------------- */

void FixImageQ2D::assign_img_charges() {
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
    if (tag_parent != 0) {
      q[ii] = -charge[abs(tag_parent)]; // use abs because tag_parent can be negative for anodes
    }
  }

  // reset KSpace charges
  if (force->kspace) force->kspace->qsum_qsq();
}

/* ---------------------------------------------------------------------- */

void FixImageQ2D::update_img_positions() {
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
//      printf("%d %d %d, %f %f %f\n", comm->me, ii, tag, xyz_local[3*tag], xyz_local[3*tag+1], xyz_local[3*tag+2]);
    }
  }
  MPI_Allreduce(*xyz_local, *xyz, (atom->natoms + 1) * 3, MPI_DOUBLE, MPI_SUM, world);

//  if (comm->me == 0){
//    for (int i = 1; i < atom->natoms + 1; i++) {
//      printf("%d %d, %f %f %f\n", i, img_parent[i], xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]);
//    }
//  }

  // update the xyz of image particles
  for (int ii = 0; ii < atom->nlocal; ii++) {
    if (mask[ii] & group2bit) {
      tag = atom->tag[ii];
      tag_parent = img_parent[tag];
        xyz_img_tmp[0] = x[ii][0];
        xyz_img_tmp[1] = x[ii][1];
        xyz_img_tmp[2] = x[ii][2];
        if (tag_parent > 0){
          x[ii][0] = xyz[tag_parent][0];
          x[ii][1] = xyz[tag_parent][1];
          x[ii][2] = 2 * z_cathode - xyz[tag_parent][2];
        }
        else if (tag_parent < 0){
          x[ii][0] = xyz[-tag_parent][0];
          x[ii][1] = xyz[-tag_parent][1];
          x[ii][2] = 2 * z_anode - xyz[-tag_parent][2];
        }
        else
          error->all(FLERR, "parent tag missing for image charge");

        // make sure the new coordinate of images is in the correct periodic box
        domain->remap_near(x[ii], xyz_img_tmp);
//      if (abs(x[ii][0] - xyz_tmp[0]) > 0.01 or abs(x[ii][1] - xyz_tmp[1]) > 0.01 or abs(x[ii][2] - xyz_tmp[2]) > 0.01)
//        printf("old and new xyz: %d %d, %f %f %f, %f %f %f\n", tag, tag_parent,
//               xyz_tmp[0], xyz_tmp[1], xyz_tmp[2], x[ii][0], x[ii][1], x[ii][2]);
    }
  }
}

void FixImageQ2D::update_electrode_charges() {
  int *mask = atom->mask;
  double **x = atom->x;
  double *q = atom->q;
  double z;
  double q_electrodes[2];
  double q_electrodes_local[2] = {0, 0};

  for (int ii = 0; ii < atom->nlocal; ii++) {
    if (mask[ii] & groupbit) {
      z = x[ii][2];
      q_electrodes_local[0] += q[ii] * (z - z_cathode) / z_span;
      q_electrodes_local[1] += q[ii] * (z_anode - z) / z_span;
    }
  }
  MPI_Allreduce(q_electrodes_local, q_electrodes, 2, MPI_DOUBLE, MPI_SUM, world);

  for (int ii = 0; ii < atom->nlocal + atom->nghost; ii++) {
    if (mask[ii] & groupbit_cathode)
      q[ii] = (q_electrodes[0] + q_voltage) / n_cathode;
    else if (mask[ii] & groupbit_anode)
      q[ii] = (q_electrodes[1] - q_voltage) / n_anode;
  }

  // reset KSpace charges
  if (force->kspace) force->kspace->qsum_qsq();
}