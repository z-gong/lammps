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
   Contributing author: Cassiano Aimoli (aimoli@gmail.com)
   Modified by: Gong Zheng (gong.zeta@gmail.com)
------------------------------------------------------------------------- */

#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "pair_mie_cut_soft.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neigh_list.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

PairMIECutSoft::PairMIECutSoft(LAMMPS *lmp) : Pair(lmp)
{
  respa_enable = 0;
  writedata = 0;
}

/* ---------------------------------------------------------------------- */

PairMIECutSoft::~PairMIECutSoft()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);
    memory->destroy(epsilon);
    memory->destroy(sigma);
    memory->destroy(gamR);
    memory->destroy(gamA);
    memory->destroy(lambda);
    memory->destroy(Cmie);
    memory->destroy(mie1);
    memory->destroy(mie2);
    memory->destroy(mie3);
    memory->destroy(mie4);
    memory->destroy(offset);
  }
}

/* ---------------------------------------------------------------------- */

void PairMIECutSoft::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double rsq,rgamA,forcemie,factor_mie;
  double denmie;
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_mie = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_mie = special_mie[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {

        rgamA = pow(rsq,gamA[itype][jtype]/2.0);
        denmie = 1.0/(mie3[itype][jtype] + rgamA*mie2[itype][jtype]);
        forcemie = mie1[itype][jtype]*gamA[itype][jtype]*rgamA*
            (mie4[itype][jtype]*pow(denmie,mie4[itype][jtype]+1)-denmie*denmie)*
            mie2[itype][jtype];
        fpair = factor_mie*forcemie/rsq;

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

        if (eflag) {
          evdwl = mie1[itype][jtype] * (pow(denmie,mie4[itype][jtype]) - denmie) -
            offset[itype][jtype];
          evdwl *= factor_mie;
        }

        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             evdwl,0.0,fpair,delx,dely,delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ---------------------------------------------------------------------- */


/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairMIECutSoft::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  memory->create(cut,n+1,n+1,"pair:cut");
  memory->create(epsilon,n+1,n+1,"pair:epsilon");
  memory->create(sigma,n+1,n+1,"pair:sigma");
  memory->create(gamR,n+1,n+1,"pair:gamR");
  memory->create(gamA,n+1,n+1,"pair:gamA");
  memory->create(lambda,n+1,n+1,"pair:lambda");
  memory->create(Cmie,n+1,n+1,"pair:Cmie");
  memory->create(mie1,n+1,n+1,"pair:mie1");
  memory->create(mie2,n+1,n+1,"pair:mie2");
  memory->create(mie3,n+1,n+1,"pair:mie3");
  memory->create(mie4,n+1,n+1,"pair:mie4");
  memory->create(offset,n+1,n+1,"pair:offset");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairMIECutSoft::settings(int narg, char **arg)
{
  if (narg != 3) error->all(FLERR,"Illegal pair_style command");

  nlambda = force->numeric(FLERR,arg[0]);
  alphalj = force->numeric(FLERR,arg[1]);
  cut_global = force->numeric(FLERR,arg[2]);

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i+1; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairMIECutSoft::coeff(int narg, char **arg)
{
  if (narg < 7 || narg > 8)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
  force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);

  double epsilon_one = force->numeric(FLERR,arg[2]);
  double sigma_one = force->numeric(FLERR,arg[3]);
  double gamR_one = force->numeric(FLERR,arg[4]);
  double gamA_one = force->numeric(FLERR,arg[5]);
  double lambda_one = force->numeric(FLERR,arg[6]);

  double cut_one = cut_global;
  if (narg == 8) cut_one = force->numeric(FLERR,arg[7]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      epsilon[i][j] = epsilon_one;
      sigma[i][j] = sigma_one;
      gamR[i][j] = gamR_one;
      gamA[i][j] = gamA_one;
      lambda[i][j] = lambda_one;
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairMIECutSoft::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    epsilon[i][j] = mix_energy(epsilon[i][i],epsilon[j][j],
                               sigma[i][i],sigma[j][j]);
    sigma[i][j] = mix_distance(sigma[i][i],sigma[j][j]);
    gamR[i][j] = mix_distance(gamR[i][i],gamR[j][j]);
    gamA[i][j] = mix_distance(gamA[i][i],gamA[j][j]);
    cut[i][j] = mix_distance(cut[i][i],cut[j][j]);
  }
  
  Cmie[i][j] = (gamR[i][j]/(gamR[i][j]-gamA[i][j]) * 
                pow((gamR[i][j]/gamA[i][j]),
                    (gamA[i][j]/(gamR[i][j]-gamA[i][j]))));
  mie1[i][j] = pow(lambda[i][j], nlambda) * Cmie[i][j] * epsilon[i][j];
  mie2[i][j] = pow(sigma[i][j], -gamA[i][j]);
  mie3[i][j] = alphalj * (1.0-lambda[i][j]) * (1.0-lambda[i][j]);
  mie4[i][j] = gamR[i][j]/gamA[i][j];

  if (offset_flag) {
    double ratio = cut[i][j] / sigma[i][j];
    double denmie = 1.0/(mie3[i][j] + pow(ratio,gamA[i][j]));
    offset[i][j] = mie1[i][j] * (pow(denmie,mie4[i][j]) - denmie);
  } else offset[i][j] = 0.0;

  gamA[j][i] = gamA[i][j];
  mie1[j][i] = mie1[i][j];
  mie2[j][i] = mie2[i][j];
  mie3[j][i] = mie3[i][j];
  mie4[j][i] = mie4[i][j];
  offset[j][i] = offset[i][j];

  // compute I,J contribution to long-range tail correction
  // count total # of atoms of type I and J via Allreduce

  if (tail_flag) {
    int *type = atom->type;
    int nlocal = atom->nlocal;

    double count[2],all[2];
    count[0] = count[1] = 0.0;
    for (int k = 0; k < nlocal; k++) {
      if (type[k] == i) count[0] += 1.0;
      if (type[k] == j) count[1] += 1.0;
    }
    MPI_Allreduce(count,all,2,MPI_DOUBLE,MPI_SUM,world);

    double siggamA = pow(sigma[i][j],gamA[i][j]);
    double siggamR = pow(sigma[i][j],gamR[i][j]);
    double rcgamA = pow(cut[i][j],(gamA[i][j]-3.0));
    double rcgamR = pow(cut[i][j],(gamR[i][j]-3.0));
    etail_ij = mie1[i][j]*2.0*MY_PI*all[0]*all[1]*(siggamR/((gamR[i][j]-3.0)*rcgamR)-siggamA/((gamA[i][j]-3.0)*rcgamA));
    ptail_ij = mie1[i][j]*2.0*MY_PI*all[0]*all[1]/3.0*
      ((gamR[i][j]/(gamR[i][j]-3.0))*siggamR/rcgamR-(gamA[i][j]/(gamA[i][j]-3.0))*siggamA/rcgamA);
  }

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairMIECutSoft::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&epsilon[i][j],sizeof(double),1,fp);
        fwrite(&sigma[i][j],sizeof(double),1,fp);
        fwrite(&gamR[i][j],sizeof(double),1,fp);
        fwrite(&gamA[i][j],sizeof(double),1,fp);
        fwrite(&lambda[i][j],sizeof(double),1,fp);
        fwrite(&cut[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairMIECutSoft::read_restart(FILE *fp)
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
          fread(&epsilon[i][j],sizeof(double),1,fp);
          fread(&sigma[i][j],sizeof(double),1,fp);
          fread(&gamR[i][j],sizeof(double),1,fp);
          fread(&gamA[i][j],sizeof(double),1,fp);
          fread(&lambda[i][j],sizeof(double),1,fp);
          fread(&cut[i][j],sizeof(double),1,fp);
        }
        MPI_Bcast(&epsilon[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&sigma[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&gamR[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&gamA[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&lambda[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairMIECutSoft::write_restart_settings(FILE *fp)
{
  fwrite(&nlambda,sizeof(double),1,fp);
  fwrite(&alphalj,sizeof(double),1,fp);

  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
  fwrite(&tail_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairMIECutSoft::read_restart_settings(FILE *fp)
{
  int me = comm->me;
  if (me == 0) {
    fread(&nlambda,sizeof(double),1,fp);
    fread(&alphalj,sizeof(double),1,fp);

    fread(&cut_global,sizeof(double),1,fp);
    fread(&offset_flag,sizeof(int),1,fp);
    fread(&mix_flag,sizeof(int),1,fp);
    fread(&tail_flag,sizeof(int),1,fp);
  }
  MPI_Bcast(&nlambda,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&alphalj,1,MPI_DOUBLE,0,world);

  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
  MPI_Bcast(&tail_flag,1,MPI_INT,0,world);
}

/* ---------------------------------------------------------------------- */

double PairMIECutSoft::single(int i, int j, int itype, int jtype, double rsq,
                           double factor_coul, double factor_mie,
                           double &fforce)
{
  double rgamA,forcemie,phimie;
  double denmie;

  rgamA = pow(rsq,gamA[itype][jtype]/2.0);
  denmie = 1.0/(mie3[itype][jtype] + rgamA*mie2[itype][jtype]);

  forcemie = mie1[itype][jtype]*gamA[itype][jtype]*rgamA*
      (mie4[itype][jtype]*pow(denmie,mie4[itype][jtype]+1)-denmie*denmie)*
      mie2[itype][jtype];
  fforce = factor_mie*forcemie/rsq;

  phimie = mie1[itype][jtype] * (pow(denmie,mie4[itype][jtype]) - denmie) -
    offset[itype][jtype];
  return factor_mie*phimie;
}

/* ---------------------------------------------------------------------- */

void *PairMIECutSoft::extract(const char *str, int &dim)
{
  dim = 2;
  if (strcmp(str,"epsilon") == 0) return (void *) epsilon;
  if (strcmp(str,"sigma") == 0) return (void *) sigma;
  if (strcmp(str,"gamR") == 0) return (void *) gamR;
  if (strcmp(str,"gamA") == 0) return (void *) gamA;
  if (strcmp(str,"lambda") == 0) return (void *) lambda;
  return NULL;
}
