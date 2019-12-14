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

#include "compute_temp_drude.h"
#include <mpi.h>
#include <cstring>
#include "atom.h"
#include "update.h"
#include "force.h"
#include "modify.h"
#include "fix_drude.h"
#include "domain.h"
#include "error.h"
#include "comm.h"
#include "memory.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeTempDrude::ComputeTempDrude(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg != 3) error->all(FLERR,"Illegal compute temp command");

  vector_flag = 1;
  scalar_flag = 1;
  size_vector = 6;

  /* ---------------------
   * change size_vector to 8
   * t_core, t_drude, t_molecule, t_internal,
   * dof_core, dof_drude, dof_molecule, dof_internal
   * ke_core, ke_drude, ke_molecule, ke_internal
   */
  size_vector = 12;
  /* ----------------------
   * end of change
   */

  extscalar = 0;
  extvector = -1;
  extlist = new int[6];
  extlist[0] = extlist[1] = 0;
  extlist[2] = extlist[3] = extlist[4] = extlist[5] = 1;
  tempflag = 0; // because does not compute a single temperature (scalar and vector)

  vector = new double[size_vector];
  fix_drude = NULL;
  id_temp = NULL;
  temperature = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeTempDrude::~ComputeTempDrude()
{
  delete [] vector;
  delete [] extlist;
  delete [] id_temp;
}

/* ---------------------------------------------------------------------- */

void ComputeTempDrude::init()
{
  int ifix;
  for (ifix = 0; ifix < modify->nfix; ifix++)
    if (strcmp(modify->fix[ifix]->style,"drude") == 0) break;
  if (ifix == modify->nfix) error->all(FLERR, "compute temp/drude requires fix drude");
  fix_drude = (FixDrude *) modify->fix[ifix];

  if (!comm->ghost_velocity)
    error->all(FLERR,"compute temp/drude requires ghost velocities. Use comm_modify vel yes");
}

/* ---------------------------------------------------------------------- */

void ComputeTempDrude::setup()
{
  dof_compute();
}

/* ---------------------------------------------------------------------- */

void ComputeTempDrude::dof_compute()
{
  int nlocal = atom->nlocal;
  int *type = atom->type;
  int dim = domain->dimension;
  int *drudetype = fix_drude->drudetype;

  fix_dof = 0;
  for (int i = 0; i < modify->nfix; i++)
    fix_dof += modify->fix[i]->dof(igroup);

  bigint dof_core_loc = 0, dof_drude_loc = 0;
  for (int i = 0; i < nlocal; i++) {
    if (atom->mask[i] & groupbit) {
      if (drudetype[type[i]] == DRUDE_TYPE) // Non-polarizable atom
          dof_drude_loc++;
      else
          dof_core_loc++;
    }
  }
  dof_core_loc *= dim;
  dof_drude_loc *= dim;
  MPI_Allreduce(&dof_core_loc,  &dof_core,  1, MPI_LMP_BIGINT, MPI_SUM, world);
  MPI_Allreduce(&dof_drude_loc, &dof_drude, 1, MPI_LMP_BIGINT, MPI_SUM, world);
  dof_core -= fix_dof;
  vector[2] = dof_core;
  vector[3] = dof_drude;

  /* ------------------------------------
   * copy from fix_tgnh.cpp
   * ------------------------------------ */
  double *mass = atom->mass;
  int *mask = atom->mask;
  int *molecule = atom->molecule;
  int n_drude, n_drude_tmp = 0;
  tagint id_mol = 0, n_mol_in_group = 0;

  for (int i = 0; i < atom->nlocal; i++) {
    // molecule id starts from 1. max(id_mol) equals to the number of molecules in the system
    id_mol = std::max(id_mol, molecule[i]);
    if (mask[i] & groupbit) {
      if (drudetype[type[i]] == DRUDE_TYPE)
        n_drude_tmp += 1;
    }
  }
  MPI_Allreduce(&n_drude_tmp, &n_drude, 1, MPI_LMP_TAGINT, MPI_SUM, world);
  MPI_Allreduce(&id_mol, &n_mol, 1, MPI_LMP_TAGINT, MPI_MAX, world);

  // use flag_mol to determin the number of molecules in the fix group
  int *flag_mol = new int[n_mol + 1];
  int *flag_mol_tmp = new int[n_mol + 1];
  memset(flag_mol_tmp, 0, sizeof(int) * (n_mol + 1));
  for (int i = 0; i < atom->nlocal; i++) {
    if (mask[i] & groupbit) {
      flag_mol_tmp[molecule[i]] = 1;
    }
  }
  MPI_Allreduce(flag_mol_tmp, flag_mol, n_mol + 1, MPI_INT, MPI_SUM, world);
  for (int i = 1; i < n_mol + 1; i++) {
    if (flag_mol[i])
      n_mol_in_group++;
  }

  // length of v_mol set to n_mol+1, so that the subscript start from 1, we can call v_mol[n_mol]
  memory->create(v_mol, n_mol + 1, 3, "compute_temp_drude::v_mol");
  memory->create(v_mol_tmp, n_mol + 1, 3, "compute_temp_drude::v_mol_tmp");
  memory->create(mass_mol, n_mol + 1, "compute_temp_drude::mass_mol");

  auto *mass_tmp = new double[n_mol + 1];
  memset(mass_tmp, 0, sizeof(double) * (n_mol + 1));
  for (int i = 0; i < atom->nlocal; i++) {
    id_mol = molecule[i];
    mass_tmp[id_mol] += mass[type[i]];
  }
  MPI_Allreduce(mass_tmp, mass_mol, n_mol + 1, MPI_DOUBLE, MPI_SUM, world);

  // DOFs
  dof_mol = 3 * n_mol_in_group;
  if (n_mol_in_group > 1)
    dof_mol -= 3; // remove DOFs of COM motion of the whole system
  dof_int = dof_core - dof_mol;
  /* ------------------------------------
   * end of copy
   * ------------------------------------ */
}

/* ---------------------------------------------------------------------- */

int ComputeTempDrude::modify_param(int narg, char **arg)
{
  if (strcmp(arg[0],"temp") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal fix_modify command");
    delete [] id_temp;
    int n = strlen(arg[1]) + 1;
    id_temp = new char[n];
    strcpy(id_temp,arg[1]);

    int icompute = modify->find_compute(id_temp);
    if (icompute < 0)
      error->all(FLERR,"Could not find fix_modify temperature ID");
    temperature = modify->compute[icompute];

    if (temperature->tempflag == 0)
      error->all(FLERR,
                 "Fix_modify temperature ID does not compute temperature");
    if (temperature->igroup != igroup && comm->me == 0)
      error->warning(FLERR,"Group for fix_modify temp != fix group");
    return 2;
  }
  return 0;
}

/* ---------------------------------------------------------------------- */

void ComputeTempDrude::compute_vector()
{
    invoked_vector = update->ntimestep;

    int nlocal = atom->nlocal;
    int *mask = atom->mask;
    int *type = atom->type;
    double *rmass = atom->rmass, *mass = atom->mass;
    double **v = atom->v;
    tagint *drudeid = fix_drude->drudeid;
    int *drudetype = fix_drude->drudetype;
    int dim = domain->dimension;
    double mvv2e = force->mvv2e, kb = force->boltz;

    double mcore, mdrude;
    double ecore, edrude;
    double *vcore, *vdrude;
    double kineng_core_loc = 0., kineng_drude_loc = 0.;
    for (int i=0; i<nlocal; i++){
        if (groupbit & mask[i] && drudetype[type[i]] != DRUDE_TYPE){
            if (drudetype[type[i]] == NOPOL_TYPE) {
                ecore = 0.;
                vcore = v[i];
                if (temperature) temperature->remove_bias(i, vcore);
                for (int k=0; k<dim; k++) ecore += vcore[k]*vcore[k];
                if (temperature) temperature->restore_bias(i, vcore);
                if (rmass) mcore = rmass[i];
                else mcore = mass[type[i]];
                kineng_core_loc += mcore * ecore;
            } else { // CORE_TYPE
                int j = atom->map(drudeid[i]);
                if (rmass) {
                    mcore = rmass[i];
                    mdrude = rmass[j];
                } else {
                    mcore = mass[type[i]];
                    mdrude = mass[type[j]];
                }
                double mtot_inv = 1. / (mcore + mdrude);
                ecore = 0.;
                edrude = 0.;
                vcore = v[i];
                vdrude = v[j];
                if (temperature) {
                    temperature->remove_bias(i, vcore);
                    temperature->remove_bias(j, vdrude);
                }
                for (int k=0; k<dim; k++) {
                    double v1 = mdrude * vdrude[k] + mcore * vcore[k];
                    ecore += v1 * v1;
                    double v2 = vdrude[k] - vcore[k];
                    edrude += v2 * v2;
                }
                if (temperature) {
                    temperature->restore_bias(i, vcore);
                    temperature->restore_bias(j, vdrude);
                }
                kineng_core_loc += mtot_inv * ecore;
                kineng_drude_loc += mtot_inv * mcore * mdrude * edrude;
            }
        }
    }

    if (dynamic) dof_compute();
    kineng_core_loc *= 0.5 * mvv2e;
    kineng_drude_loc *= 0.5 * mvv2e;
    MPI_Allreduce(&kineng_core_loc,&kineng_core,1,MPI_DOUBLE,MPI_SUM,world);
    MPI_Allreduce(&kineng_drude_loc,&kineng_drude,1,MPI_DOUBLE,MPI_SUM,world);
    temp_core = 2.0 * kineng_core / (dof_core * kb);
    temp_drude = 2.0 * kineng_drude / (dof_drude * kb);
    vector[0] = temp_core;
    vector[1] = temp_drude;
    vector[4] = kineng_core;
    vector[5] = kineng_drude;

    /* --------------------------------
     * change vector output
     */
    compute_temp_mol_int_drude();
    vector[0] = temp_core;
    vector[1] = temp_drude;
    vector[2] = t_mol;
    vector[3] = t_int;
    vector[4] = dof_core;
    vector[5] = dof_drude;
    vector[6] = dof_mol;
    vector[7] = dof_int;
    vector[8] = kineng_core;
    vector[9] = kineng_drude;
    vector[10] = ke2mol / 2;
    vector[11] = ke2int / 2;
  /* ---------------------------------
   * end of change
   */
}

double ComputeTempDrude::compute_scalar(){
    compute_vector();
    scalar = vector[0];
    return scalar;
}

/* ------------------------------------
 * copy from fix_tgnh.cpp
 * ------------------------------------ */
void ComputeTempDrude::compute_temp_mol_int_drude() {
  double **v = atom->v;
  double *mass = atom->mass;
  int *molecule = atom->molecule;
  int *type = atom->type;
  int *mask = atom->mask;
  int *drudetype = fix_drude->drudetype;
  int *drudeid = fix_drude->drudeid;
  int imol, ci, di;
  double mass_com, mass_reduced, mass_core, mass_drude;
  double vint, vcom, vrel;
  double ke2_int_tmp = 0, ke2_drude_tmp = 0;
  double boltz = force->boltz;

  memset(*v_mol_tmp, 0, sizeof(double) * (n_mol + 1) * 3); // the length of v_mol is n_mol+1

  for (int i = 0; i < atom->nlocal; i++) {
    if (mask[i] & groupbit) {
      imol = molecule[i];
      for (int k = 0; k < 3; k++)
        v_mol_tmp[imol][k] += v[i][k] * mass[type[i]];
    }
  }
  MPI_Allreduce(*v_mol_tmp, *v_mol, (n_mol + 1) * 3, MPI_DOUBLE, MPI_SUM, world);

  ke2mol = 0;
  for (int i = 1; i < n_mol + 1; i++) {
    for (int k = 0; k < 3; k++) {
      v_mol[i][k] /= mass_mol[i];
      ke2mol += mass_mol[i] * (v_mol[i][k] * v_mol[i][k]);
    }
  }
  ke2mol *= force->mvv2e;
  t_mol = ke2mol / dof_mol / boltz;

  for (int i = 0; i < atom->nlocal; i++) {
    if (mask[i] & groupbit) {
      imol = molecule[i];
      if (drudetype[type[i]] == NOPOL_TYPE) {
        for (int k = 0; k < 3; k++) {
          vint = v[i][k] - v_mol[imol][k];
          ke2_int_tmp += mass[type[i]] * vint * vint;
        }
      } else if (drudetype[type[i]] == CORE_TYPE) {
        di = atom->map(drudeid[i]);
        mass_core = mass[type[i]];
        mass_drude = mass[type[di]];
        mass_com = mass_core + mass_drude;
        mass_reduced = mass_core * mass_drude / mass_com;
        for (int k = 0; k < 3; k++) {
          vcom = (mass_core * v[i][k] + mass_drude * v[di][k]) / mass_com;
          vint = vcom - v_mol[imol][k];
          ke2_int_tmp += mass_com * vint * vint;
          vrel = v[di][k] - v[i][k];
          ke2_drude_tmp += mass_reduced * vrel * vrel;
        }
      }
    }
  }
  MPI_Allreduce(&ke2_int_tmp, &ke2int, 1, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(&ke2_drude_tmp, &ke2drude, 1, MPI_DOUBLE, MPI_SUM, world);
  ke2int *= force->mvv2e;
  ke2drude *= force->mvv2e;
  t_int = ke2int / dof_int / boltz;
  t_drude = ke2drude / dof_drude / boltz;
}
/* ------------------------------------
 * end of copy
 * ------------------------------------ */
