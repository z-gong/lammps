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

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "fix_chargetransfer.h"
#include "atom.h"
#include "update.h"
#include "group.h"
#include "modify.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "neigh_list.h"
#include "pair.h"
#include "pair_hybrid.h"
#include "kspace.h"
#include "fix_store.h"
#include "input.h"
#include "variable.h"
#include "respa.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

enum {
    PAIR, KSPACE, ATOM
};
enum {
    DIAMETER, CHARGE
};

/* ---------------------------------------------------------------------- */

FixCharge::FixCharge(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg) {
    if (narg < 5) error->all(FLERR, "Illegal fix charge command");
    nevery = force->inumeric(FLERR, arg[3]);
    if (nevery < 0) error->all(FLERR, "Illegal fix charge command");

    dynamic_group_allow = 1;
    create_attribute = 1;

    // count # of adaptations

    nadapt = 0;

    int iarg = 4;
    while (iarg < narg) {
        if (iarg + 5 > narg) error->all(FLERR, "Illegal fix charge command");
        nadapt++;
        iarg += 5;
    }

    if (nadapt == 0) error->all(FLERR, "Illegal fix charge command");
    adapt = new Adapt[nadapt];

    // parse keywords

    nadapt = 0;

    iarg = 4;
    while (iarg < narg) {
        if (iarg + 5 > narg) error->all(FLERR, "Illegal fix charge command");
        adapt[nadapt].atype1 = force->inumeric(FLERR, arg[iarg]);
        adapt[nadapt].atype2 = force->inumeric(FLERR, arg[iarg + 1]);
        adapt[nadapt].dq = force->numeric(FLERR, arg[iarg + 2]);
        double rc = force->numeric(FLERR, arg[iarg + 3]);
        adapt[nadapt].sigma = force->numeric(FLERR, arg[iarg + 4]);
        adapt[nadapt].rc = rc;
        adapt[nadapt].cutsq = rc * rc;
        nadapt++;
        iarg += 5;
    }

    id_fix_chg = NULL;
}

/* ---------------------------------------------------------------------- */

FixCharge::~FixCharge() {
    delete[] adapt;

    // check nfix in case all fixes have already been deleted

    if (id_fix_chg && modify->nfix) modify->delete_fix(id_fix_chg);
    delete[] id_fix_chg;
}

/* ---------------------------------------------------------------------- */

int FixCharge::setmask() {
    int mask = 0;
    mask |= PRE_FORCE;
    mask |= POST_RUN;
    mask |= PRE_FORCE_RESPA;
    return mask;
}

/* ----------------------------------------------------------------------
   if need to restore per-atom quantities, create new fix STORE styles
------------------------------------------------------------------------- */

void FixCharge::post_constructor() {
    id_fix_chg = NULL;

    char **newarg = new char *[6];
    newarg[1] = group->names[igroup];
    newarg[2] = (char *) "STORE";
    newarg[3] = (char *) "peratom";
    newarg[4] = (char *) "1";
    newarg[5] = (char *) "1";

    int n = strlen(id) + strlen("_FIX_STORE_CHG") + 1;
    id_fix_chg = new char[n];
    strcpy(id_fix_chg, id);
    strcat(id_fix_chg, "_FIX_STORE_CHG");
    newarg[0] = id_fix_chg;
    modify->add_fix(6, newarg);
    fix_chg = (FixStore *) modify->fix[modify->nfix - 1];

    if (fix_chg->restart_reset) fix_chg->restart_reset = 0;
    else {
        double *vec = fix_chg->vstore;
        double *q = atom->q;
        int *mask = atom->mask;
        int nlocal = atom->nlocal;

        for (int i = 0; i < nlocal; i++) {
            if (mask[i] & groupbit) vec[i] = q[i];
            else vec[i] = 0.0;
        }
    }

    delete[]newarg;
}

/* ---------------------------------------------------------------------- */

void FixCharge::init() {
    // allow a dynamic group only if ATOM attribute not used
    if (group->dynamic[igroup])
        error->all(FLERR, "Cannot use dynamic group with fix charge/transfer atom");

    // setup and error checks
    if (!atom->q_flag)
        error->all(FLERR, "Fix adapt requires atom attribute charge");

    // fixes that store initial per-atom values
    if (id_fix_chg) {
        int ifix = modify->find_fix(id_fix_chg);
        if (ifix < 0) error->all(FLERR, "Could not find fix charge/transfer storage fix ID");
        fix_chg = (FixStore *) modify->fix[ifix];
    }

    if (strstr(update->integrate_style, "respa"))
        nlevels_respa = ((Respa *) update->integrate)->nlevels;

    // need a full neighbor list, built every Nevery steps
    int irequest = neighbor->request(this, instance_me);
    neighbor->requests[irequest]->pair = 0;
    neighbor->requests[irequest]->fix = 1;
    neighbor->requests[irequest]->half = 0;
    neighbor->requests[irequest]->full = 1;
    neighbor->requests[irequest]->occasional = 1;
}

void FixCharge::init_list(int id, NeighList *ptr) {
    list = ptr;
}

/* ---------------------------------------------------------------------- */

void FixCharge::setup_pre_force(int vflag) {
    charge_transfer();
}

/* ---------------------------------------------------------------------- */

void FixCharge::setup_pre_force_respa(int vflag, int ilevel) {
    if (ilevel < nlevels_respa - 1) return;
    setup_pre_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixCharge::pre_neighbor(int vflag) {
}

void FixCharge::pre_force(int vflag) {
    if (nevery == 0) return;
    if (update->ntimestep % nevery) return;
    charge_transfer();
}

/* ---------------------------------------------------------------------- */

void FixCharge::pre_force_respa(int vflag, int ilevel, int) {
    if (ilevel < nlevels_respa - 1) return;
    pre_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixCharge::post_run() {
    restore_charge();
    // reset KSpace charges if charges have changed
    if (force->kspace) force->kspace->qsum_qsq();
}

/* ----------------------------------------------------------------------
   change pair,kspace,atom parameters based on variable evaluation
------------------------------------------------------------------------- */
void FixCharge::charge_transfer() {
    restore_charge();

    int i, j, ii, jj, inum, jnum, itype, jtype;
    double xtmp, ytmp, ztmp, delx, dely, delz, rsq, r;
    int *ilist, *jlist, *numneigh, **firstneigh;

    double **x = atom->x;
    int *tag = atom->tag;
    int *type = atom->type;
    int *molecule = atom->molecule;
    double *q = atom->q;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    int nghost = atom->nghost;
    int nall = nlocal + nghost;

    neighbor->build_one(list, 1);
    inum = list->inum;
    ilist = list->ilist;
    numneigh = list->numneigh;
    firstneigh = list->firstneigh;

    for (ii = 0; ii < inum; ii++) {
        i = ilist[ii];
        if (!(mask[i] & groupbit)) continue;
        xtmp = x[i][0];
        ytmp = x[i][1];
        ztmp = x[i][2];
        itype = type[i];
        jlist = firstneigh[i];
        jnum = numneigh[i];

        for (jj = 0; jj < jnum; jj++) {
            j = jlist[jj];
            j &= NEIGHMASK;
            if (!(mask[j] & groupbit)) continue;
            if (molecule[i] == molecule[j]) continue;
            jtype = type[j];
            delx = xtmp - x[j][0];
            dely = ytmp - x[j][1];
            delz = ztmp - x[j][2];
            rsq = delx * delx + dely * dely + delz * delz;

            for (int m = 0; m < nadapt; m++) {
                Adapt *ad = &adapt[m];
                if (rsq > ad->cutsq) continue;

                int direction = 0;
                if (itype == ad->atype1 && jtype == ad->atype2)
                    direction = 1;
                else if (itype == ad->atype2 && jtype == ad->atype1)
                    direction = -1;

                if (direction != 0) {
                    double dq = ad->dq;
                    r = sqrt(rsq);
                    if (ad->sigma > 0 && r > ad->rc - ad->sigma)
                        dq = ad->dq * 0.5 * (1 + cos((r - ad->rc + ad->sigma) / ad->sigma * 3.1415926535));

                    q[ii] += dq * direction;
                }
            }
        }
    }

    int n_atoms;
    MPI_Allreduce(&nlocal, &n_atoms, 1, MPI_INT, MPI_SUM, world);

    double *tag_q = new double[n_atoms + 1];
    memset(tag_q, 0, (n_atoms + 1) * sizeof(double));
    double *all_q = new double[n_atoms + 1];

    for (i = 0; i < nlocal; i++) {
        tag_q[tag[i]] = q[i];
    }
    MPI_Allreduce(tag_q, all_q, n_atoms + 1, MPI_DOUBLE, MPI_SUM, world);

    //if (me == 0) {
    //    for (int itag = 0; itag < n_atoms; itag++)
    //        printf("%d %f \n", itag, all_q[itag+1]);
    //}

    for (i = 0; i < nall; i++) {
        q[i] = all_q[tag[i]];
    }

    delete[] tag_q;
    delete[] all_q;

    // reset KSpace charges if charges have changed
    if (force->kspace) force->kspace->qsum_qsq();
}

/* ----------------------------------------------------------------------
   restore pair,kspace,atom parameters to original values
------------------------------------------------------------------------- */

void FixCharge::restore_charge() {
    double *vec = fix_chg->vstore;
    double *q = atom->q;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    for (int i = 0; i < nlocal; i++)
        if (mask[i] & groupbit) q[i] = vec[i];
}

/* ----------------------------------------------------------------------
   initialize one atom's storage values, called when atom is created
------------------------------------------------------------------------- */

void FixCharge::set_arrays(int i) {
    if (fix_chg) fix_chg->vstore[i] = atom->q[i];
}
