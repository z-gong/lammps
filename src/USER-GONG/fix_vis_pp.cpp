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

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixVisPP::FixVisPP(LAMMPS *lmp, int narg, char **arg) :
        Fix(lmp, narg, arg),
        xstr(NULL), ystr(NULL), zstr(NULL), idregion(NULL), sforce(NULL) {
    if (narg < 4) error->all(FLERR, "Illegal fix setforce command");

    dynamic_group_allow = 1;
    vector_flag         = 1;
    size_vector         = 3;
    global_freq         = 1;
    extvector           = 1;
    respa_level_support = 1;
    ilevel_respa        = nlevels_respa = 0;
    xstr                = ystr          = zstr = NULL;

    xvalue = force->numeric(FLERR, arg[3]);
    // optional args

    iregion  = -1;
    idregion = NULL;

    int iarg = 4;
    while (iarg < narg) {
        if (strcmp(arg[iarg], "region") == 0) {
            if (iarg + 2 > narg) error->all(FLERR, "Illegal fix setforce command");
            iregion = domain->find_region(arg[iarg + 1]);
            if (iregion == -1)
                error->all(FLERR, "Region ID for fix setforce does not exist");
            int n   = strlen(arg[iarg + 1]) + 1;
            idregion = new char[n];
            strcpy(idregion, arg[iarg + 1]);
            iarg += 2;
        } else error->all(FLERR, "Illegal fix setforce command");
    }

    maxatom = 1;
    memory->create(sforce, maxatom, 3, "setforce:sforce");
}

/* ---------------------------------------------------------------------- */

FixVisPP::~FixVisPP() {
    if (copymode) return;

    delete[] xstr;
    delete[] ystr;
    delete[] zstr;
    delete[] idregion;
    memory->destroy(sforce);
}

/* ---------------------------------------------------------------------- */

int FixVisPP::setmask() {
    int mask = 0;
    mask |= POST_FORCE;
    mask |= POST_FORCE_RESPA;
    mask |= MIN_POST_FORCE;
    return mask;
}

/* ---------------------------------------------------------------------- */

void FixVisPP::init() {
    // set index and check validity of region

    if (iregion >= 0) {
        iregion = domain->find_region(idregion);
        if (iregion == -1)
            error->all(FLERR, "Region ID for fix ppm does not exist");
    }

    if (strstr(update->integrate_style, "respa")) {
        nlevels_respa = ((Respa *) update->integrate)->nlevels;
        if (respa_level >= 0) ilevel_respa = MIN(respa_level, nlevels_respa - 1);
        else ilevel_respa = nlevels_respa - 1;
    }

    // cannot use non-zero forces for a minimization since no energy is integrated
    // use fix addforce instead

    int flag = 0;
    if (update->whichflag == 2) {
        error->all(FLERR, "Cannot use non-zero forces in an energy minimization");
    }
}

/* ---------------------------------------------------------------------- */

void FixVisPP::setup(int vflag) {
    if (strstr(update->integrate_style, "verlet"))
        post_force(vflag);
    else
        for (int ilevel = 0; ilevel < nlevels_respa; ilevel++) {
            ((Respa *) update->integrate)->copy_flevel_f(ilevel);
            post_force_respa(vflag, ilevel, 0);
            ((Respa *) update->integrate)->copy_f_flevel(ilevel);
        }
}

/* ---------------------------------------------------------------------- */

void FixVisPP::min_setup(int vflag) {
    post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixVisPP::post_force(int vflag) {
    double **x    = atom->x;
    double **f    = atom->f;
    int    *type  = atom->type;
    double *mass  = atom->mass;
    double *rmass = atom->rmass;
    int    *mask  = atom->mask;
    int    nlocal = atom->nlocal;

    double massone, force_x, acc_x;
    double zlo    = domain->boxlo[2];
    double zhi    = domain->boxhi[2];

    // update region if necessary

    Region *region = NULL;
    if (iregion >= 0) {
        region = domain->regions[iregion];
        region->prematch();
    }

    for (int i = 0; i < nlocal; i++)
        if (mask[i] & groupbit) {
            if (rmass) massone = rmass[i];
            else massone = mass[type[i]];

            if (region && !region->match(x[i][0], x[i][1], x[i][2])) continue;

            acc_x   = xvalue * cos(2 * 3.1415926535 * (x[i][2] - zlo) / (zhi - zlo));
            force_x = acc_x * massone * force->mvv2e; // unit from (g/mol)*(A/fs^2) to kcal/(mol.A)
//      printf("%f %f %f %f\n", mass[type[i]], x[i][2], acc_x, force_x);

            f[i][0] += force_x;
        }
}

/* ---------------------------------------------------------------------- */

void FixVisPP::post_force_respa(int vflag, int ilevel, int iloop) {
    // set force to desired value on requested level, 0.0 on other levels

    if (ilevel == ilevel_respa) post_force(vflag);
    else {
        Region *region = NULL;
        if (iregion >= 0) {
            region = domain->regions[iregion];
            region->prematch();
        }

        double **x    = atom->x;
        double **f    = atom->f;
        int    *mask  = atom->mask;
        int    nlocal = atom->nlocal;

        for (int i = 0; i < nlocal; i++)
            if (mask[i] & groupbit) {
                if (region && !region->match(x[i][0], x[i][1], x[i][2])) continue;
                if (xstyle) f[i][0] = 0.0;
                if (ystyle) f[i][1] = 0.0;
                if (zstyle) f[i][2] = 0.0;
            }
    }
}

/* ---------------------------------------------------------------------- */

void FixVisPP::min_post_force(int vflag) {
    post_force(vflag);
}

/* ----------------------------------------------------------------------
   return components of total force on fix group before force was changed
------------------------------------------------------------------------- */

double FixVisPP::compute_vector(int n) {
    return 0;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double FixVisPP::memory_usage() {
    return 0;
}
