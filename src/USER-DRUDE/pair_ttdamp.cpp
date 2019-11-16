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

#include "pair_ttdamp.h"
#include <mpi.h>
#include <cmath>
#include <cstring>
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"
#include "fix.h"
#include "fix_drude.h"
#include "domain.h"
#include "modify.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairTTDamp::PairTTDamp(LAMMPS *lmp) : Pair(lmp) {
    fix_drude = NULL;
}

/* ---------------------------------------------------------------------- */

PairTTDamp::~PairTTDamp()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(b);
    memory->destroy(c);
    memory->destroy(ntt);
    memory->destroy(cut);
    memory->destroy(scale);
  }
}

/* ---------------------------------------------------------------------- */

void PairTTDamp::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double qi,qj,xtmp,ytmp,ztmp,delx,dely,delz,ecoul,fpair;
  double r,rsq,r2inv,rinv,factor_coul;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double qq, dEdr;
  int di,dj;
  double alpha, alphaprime, beta, gamma, betaprime, gammaprime, gammatmp;

  ecoul = 0.0;
  ev_init(eflag,vflag);

  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_coul = force->special_coul;
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e;
  int *drudetype = fix_drude->drudetype;
  tagint *drudeid = fix_drude->drudeid;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];

    qi = q[i];
    // get dq of the core via the drude charge
    if (drudetype[type[i]] == CORE_TYPE) {
      di = domain->closest_image(i, atom->map(drudeid[i]));
      qi = -q[di];
    }

    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;

      // only on polar-nonpolar pair
      if (drudetype[type[i]] == NOPOL_TYPE && drudetype[type[j]] == NOPOL_TYPE)
        continue;
      if (drudetype[type[i]] != NOPOL_TYPE && drudetype[type[j]] != NOPOL_TYPE)
        continue;

      qj = q[j];
      // get dq of the core via the drude charge
      if (drudetype[type[j]] == CORE_TYPE) {
        dj = domain->closest_image(j, atom->map(drudeid[j]));
        qj = -q[dj];
      }

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r2inv = 1.0/rsq;
        rinv = sqrt(r2inv);

        r = sqrt(rsq);
        alpha = rinv;
        alphaprime = -r2inv;
        beta = c[itype][jtype] * exp(-b[itype][jtype] * r);
        betaprime = -b[itype][jtype] * beta;
        gamma = 1;
        gammaprime = 0;
        for (int k = 1; k <= ntt[itype][jtype]; k++) {
          gammatmp = pow(b[itype][jtype] * r, k - 1) / factorial[k];
          gamma += gammatmp * b[itype][jtype] * r;
          gammaprime += gammatmp * b[itype][jtype] * k;
        }
        qq = qqrd2e * qi * qj * scale[itype][jtype];
        dEdr = -(alphaprime * beta * gamma + alpha * betaprime * gamma + alpha * beta * gammaprime) * qq;
        fpair = -rinv * dEdr * factor_coul;

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

        if (eflag)
          ecoul = -alpha * beta * gamma * qq * factor_coul;

        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             0.0,ecoul,fpair,delx,dely,delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairTTDamp::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  memory->create(cut,n+1,n+1,"pair:cut");
  memory->create(scale,n+1,n+1,"pair:scale");
  memory->create(b, n + 1, n + 1, "pair:b");
  memory->create(c, n + 1, n + 1, "pair:c");
  memory->create(ntt, n + 1, n + 1, "pair:ntt");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairTTDamp::settings(int narg, char **arg)
{
  if (narg != 2) error->all(FLERR, "Illegal pair_style command");

  n_global = force->inumeric(FLERR, arg[0]);
  cut_global = force->numeric(FLERR, arg[1]);

  factorial.resize(n_global+1);
  factorial[0] = 1;
  for (int i = 1; i <= n_global; i++)
    factorial[i] = i * factorial[i - 1];

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) {
            ntt[i][j] = n_global;
          cut[i][j] = cut_global;
        }
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairTTDamp::coeff(int narg, char **arg)
{
  if (narg < 3 || narg > 5)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
  force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);

  double b_one = force->numeric(FLERR, arg[2]);
  double c_one = force->numeric(FLERR, arg[3]);
  int n_one = n_global;
  double cut_one = cut_global;
  if (narg >= 5) n_one = force->inumeric(FLERR,arg[4]);
  if (narg == 6) cut_one = force->numeric(FLERR,arg[5]);

  if (n_one > n_global)
    error->all(FLERR, "Incorrect coefficients for pair style ttdamp: n should not be larger than global setting");

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      b[i][j] = b_one;
      c[i][j] = c_one;
      ntt[i][j] = n_one;
      cut[i][j] = cut_one;
      scale[i][j] = 1.0;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}


/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairTTDamp::init_style()
{
  if (!atom->q_flag)
    error->all(FLERR,"Pair style ttdamp requires atom attribute q");
  int ifix;
  for (ifix = 0; ifix < modify->nfix; ifix++)
    if (strcmp(modify->fix[ifix]->style,"drude") == 0) break;
  if (ifix == modify->nfix) error->all(FLERR, "Pair ttdamp requires fix drude");
  fix_drude = (FixDrude *) modify->fix[ifix];

  neighbor->request(this,instance_me);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairTTDamp::init_one(int i, int j)
{
  if (setflag[i][j] == 0)
    cut[i][j] = mix_distance(cut[i][i], cut[j][j]);

  b[j][i] = b[i][j];
  c[j][i] = c[i][j];
  ntt[j][i] = ntt[i][j];
  scale[j][i] = scale[i][j];

  return cut[i][j];
}

/* ----------------------------------------------------------------------
  proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairTTDamp::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&b[i][j], sizeof(double), 1, fp);
        fwrite(&c[i][j], sizeof(double), 1, fp);
        fwrite(&ntt[i][j], sizeof(double), 1, fp);
        fwrite(&cut[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
  proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairTTDamp::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) fread(&setflag[i][j],sizeof(int),1,fp);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          fread(&b[i][j], sizeof(double), 1, fp);
          fread(&c[i][j], sizeof(double), 1, fp);
          fread(&ntt[i][j], sizeof(double), 1, fp);
          fread(&cut[i][j],sizeof(double),1, fp);
          }
        MPI_Bcast(&b[i][j],  1, MPI_DOUBLE,0,world);
        MPI_Bcast(&c[i][j],  1, MPI_DOUBLE,0,world);
        MPI_Bcast(&ntt[i][j],  1, MPI_DOUBLE,0,world);
        MPI_Bcast(&cut[i][j],1, MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
  proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairTTDamp::write_restart_settings(FILE *fp)
{
  fwrite(&n_global, sizeof(double), 1, fp);
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
  proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairTTDamp::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    fread(&n_global, sizeof(double), 1, fp);
    fread(&cut_global,sizeof(double),1,fp);
    fread(&offset_flag,sizeof(int),1,fp);
    fread(&mix_flag,sizeof(int),1,fp);
  }
  MPI_Bcast(&n_global, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
}

/* ---------------------------------------------------------------------- */

double PairTTDamp::single(int i, int j, int itype, int jtype,
                         double rsq, double factor_coul, double /*factor_lj*/,
                         double &fforce)
{
  double r2inv,rinv,r,phicoul;
  double qi,qj,dEdr,qq;
  double alpha, alphaprime, beta, betaprime, gamma, gammaprime, gammatmp;
  int di, dj;

  // Should not use fix_drude to determine Drude pairs in single() function
  // It will not work

//  int *drudetype = fix_drude->drudetype;
//  tagint *drudeid = fix_drude->drudeid;

  // only on polar-nonpolar pair
//  if (drudetype[itype] == NOPOL_TYPE && drudetype[jtype] == NOPOL_TYPE)
//    return 0.0;
//  if (drudetype[itype] != NOPOL_TYPE && drudetype[jtype] != NOPOL_TYPE)
//    return 0.0;

  qi = atom->q[i];
  // get dq of the core via the drude charge
//  if (drudetype[itype] == CORE_TYPE) {
//    di = domain->closest_image(i, atom->map(drudeid[i]));
//    qi = -atom->q[di];
//  }

  qj = atom->q[j];
  // get dq of the core via the drude charge
//  if (drudetype[jtype] == CORE_TYPE) {
//    dj = domain->closest_image(j, atom->map(drudeid[j]));
//    qj = -atom->q[dj];
//  }

  r2inv = 1.0/rsq;
  fforce = phicoul = 0.0;
  if (rsq < cutsq[itype][jtype]) {
    rinv = sqrt(r2inv);
    r = sqrt(rsq);
    alpha = rinv;
    alphaprime = -r2inv;
    beta = c[itype][jtype] * exp(-b[itype][jtype] * r);
    betaprime = -b[itype][jtype] * beta;
    gamma = 1;
    gammaprime = 0;
    for (int k = 1; k <= ntt[itype][jtype]; k++) {
      gammatmp = pow(b[itype][jtype] * r, k - 1) / factorial[k];
      gamma += gammatmp * b[itype][jtype] * r;
      gammaprime += gammatmp * b[itype][jtype] * k;
    }
    qq = force->qqrd2e * qi * qj * scale[itype][jtype];
    dEdr = -(alphaprime * beta * gamma + alpha * betaprime * gamma + alpha * beta * gammaprime) * qq;
    fforce = -rinv * dEdr * factor_coul;
    phicoul = -alpha * beta * gamma * qq * factor_coul;
  }

  return phicoul;
}

/* ---------------------------------------------------------------------- */

void *PairTTDamp::extract(const char *str, int &dim)
{
  dim = 2;
  if (strcmp(str,"scale") == 0) return (void *) scale;
  if (strcmp(str,"b") == 0) return (void *) b;
  if (strcmp(str,"c") == 0) return (void *) c;
  if (strcmp(str,"ntt") == 0) return (void *) ntt;
  return NULL;
}
