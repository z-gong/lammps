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

#include "mpi.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "domain.h"
#include "atom.h"
#include "update.h"
#include "group.h"
#include "force.h"
#include "comm.h"
#include "memory.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "error.h"

#include "style_pair.h"
#include "style_kspace.h"

#include "compute_inter.h"

#define MAX_GROUP 32

using namespace LAMMPS_NS;
#define EWALD_F   1.12837917
#define EWALD_P   0.3275911
#define A1        0.254829592
#define A2       -0.284496736
#define A3        1.421413741
#define A4       -1.453152027
#define A5        1.061405429

/* ----------------------------------------------------------------------
   construct the compute. group shoud be allocated before.
   ---------------------------------------------------------------------- */

ComputeInter::ComputeInter(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  MPI_Comm_rank(world,&me);

  if (! atom->molecular) error->all(FLERR,"Compute inter is valid only for molecular systems");
  if (igroup != 0) error->all(FLERR,"Compute inter group-ID should be all (0)");

  scalar_flag=0;
  vector_flag=1;
  size_vector=2;
  extscalar=0;
  extvector=1;

  vector = new double[6];
  double eng_pair_intra;

  ignore_tail = false;
  if (narg > 5) error->all(FLERR,"Compute inter has too many arguments");
  if (narg == 5){
    if (strcmp(arg[3],"tail")!=0) error->all(FLERR, "compute inter argument unknown");
    else if(strcmp(arg[4],"no")==0) ignore_tail = true;
  }
}

/* ---------------------------------------------------------------------- */

ComputeInter::~ComputeInter()
{
  delete [] vector;
}

/* ---------------------------------------------------------------------- */

void ComputeInter::init()
{
  //SINGLE_MODE mode = NORMAL;
  //if ( strcmp(force->pair_style,"lj/cut/tip4p/cut") || strcmp(force->pair_style,"lj/cut/tip4p/long"))
  //{
    //mode = TIP4P;
  //}
}

/* ---------------------------------------------------------------------- */

int ComputeInter::pack_comm(int n, int *list, double *buf, int pbc_flag, int *pbc)
{
  int i, j, m, t;
  m = 0;
  if ( pbc_flag == 0 ) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++]=atom->image[j];
      buf[m++]=atom->molecule[j];
    }
  } else {
    for (i = 0; i < n; i++) {
      j = list[i];
      t = ( ( atom->image[j]        & 1023)-pbc[0])     |
          ((((atom->image[j] >> 10) & 1023)-pbc[1])<<10)|
          ((((atom->image[j] >> 20) & 1023)-pbc[2])<<20);
      buf[m++]=t;
      buf[m++]=atom->molecule[j];
    }
  }
  return(2);
}

/* ---------------------------------------------------------------------- */

void ComputeInter::unpack_comm(int n, int first, double *buf)
{
  int i, m, last;
  m = 0;
  last = first + n;
  for (i = first; i < last; i++)
  {
    atom->image[i] = static_cast<int> (buf[m++]);
    atom->molecule[i] = static_cast<int> (buf[m++]);
  }
}

/* ---------------------------------------------------------------------- */

void ComputeInter::compute_vector()
{
  // Remap atom in order to update image array.

  invoked_vector = update->ntimestep;

  comm->forward_comm_compute(this);

  vector[0] = vector[1] = 0.0;
  eng_pair_intra = 0.0;

  compute_intra();

  // copy thermo.cpp
  double dvalue;
  double tmp = 0.0;
  if (force->pair) tmp += force->pair->eng_vdwl + force->pair->eng_coul;
  MPI_Allreduce(&tmp,&dvalue,1,MPI_DOUBLE,MPI_SUM,world);
  if (force->kspace) dvalue += force->kspace->energy;
  if (force->pair && force->pair->tail_flag && !ignore_tail) {
    double volume = domain->xprd * domain->yprd * domain->zprd;
    dvalue += force->pair->etail / volume;
  }

  // Gather data.
  MPI_Allreduce(&(eng_pair_intra),&(vector[1]),1,MPI_DOUBLE,MPI_SUM,world);
  vector[0] = dvalue - vector[1];
}

/* ---------------------------------------------------------------------- */

double ComputeInter::memory_usage()
{
  int i;
  double m=0.0;
  m += sizeof(double);     // eng_pair_intra
  m += 2*sizeof(double);     // vector
  return(m);
}


/* ---------------------------------------------------------------------- */

void ComputeInter::compute_intra()
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double xtmp,ytmp,ztmp,delx,dely,delz;
  double epair,factor_coul,factor_lj;
  double rsq;

  int *molecule=atom->molecule;

  double **x = atom->x;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  double *special_coul = force->special_coul;
  double *special_lj = force->special_lj;
  Pair *pair = force->pair;
  NeighList *list = force->pair->list;
  int newton_pair = force->newton_pair;

  double factor;
  double tmp;

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

      factor_lj = special_lj[sbmask(j)];
      factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      //if ( molecule[i] == molecule[j] && mode == NORMAL ){
      if ( molecule[i] == molecule[j]){
        epair = pair->single(i,j,itype,jtype,rsq,factor_coul,factor_lj,tmp);
        if ( force->kspace ) epair += coul_kspace_correct(i,j,itype,jtype,rsq,factor_coul);
      }
      else
        epair = 0.0;

      factor = ( newton_pair || (i < nlocal && j < nlocal) ) ? 1.0 : 0.5;
      eng_pair_intra += factor*epair;
    }
  }
  //printf("%f\n", eng_pair_intra);
}


double ComputeInter::coul_kspace_correct(int i, int j,int itype,int jtype,double rsq,double factor_coul)
{
  // copy from pair_coul_long.cpp
  double r2inv,r,grij,expm2,t,erfc,prefactor;
  double fraction,table,phicoul;
  double g_ewald = force->kspace->g_ewald;

  r2inv = 1.0/rsq;
  //if (!ncoultablebits || rsq <= tabinnersq) {
    r = sqrt(rsq);
    grij = g_ewald * r;
    expm2 = exp(-grij*grij);
    t = 1.0 / (1.0 + EWALD_P*grij);
    erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
    prefactor = force->qqrd2e * atom->q[i]*atom->q[j]/r;
  //} else {
    //union_int_float_t rsq_lookup;
    //rsq_lookup.f = rsq;
    //itable = rsq_lookup.i & ncoulmask;
    //itable >>= ncoulshiftbits;
    //fraction = (rsq_lookup.f - rtable[itable]) * drtable[itable];
    //table = ftable[itable] + fraction*dftable[itable];
    //if (factor_coul < 1.0) {
      //table = ctable[itable] + fraction*dctable[itable];
      //prefactor = atom->q[i]*atom->q[j] * table;
    //}
  //}

  //if (!ncoultablebits || rsq <= tabinnersq)
    phicoul = prefactor*erfc;
  //else {
    //table = etable[itable] + fraction*detable[itable];
    //phicoul = atom->q[i]*atom->q[j] * table;
  //}
  if (factor_coul < 1.0) phicoul -= (1.0-factor_coul)*prefactor;

  //copy from pair_coul_cut.cpp
  double rinv,coul_short_range;
  rinv = sqrt(r2inv);
  coul_short_range = force->qqrd2e * atom->q[i]*atom->q[j]*rinv* factor_coul;
  return coul_short_range - phicoul;
}

