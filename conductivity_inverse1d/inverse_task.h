#ifndef INVERSE_TASK_H_SENTRY
#define INVERSE_TASK_H_SENTRY

#include "direct_task.h"

struct paramvec
{
   vector<double> sigmas;
   vector<double> heights;
};

struct inverse_task_data
{
   paramvec init_appr, reg_vec;

   double acc_r, acc_z;
};

void inverse_task_init(inverse_task_data& inv_struct, direct_task_data& experiment_info);

void inverse_task_solve(inverse_task_data& inv_struct, vector<double>& synthetic_eds, mesh& carcass,
   direct_task_data& dtd_synth, direct_task_data& dtd_prim);
void dtd_init_from_paramvec(direct_task_data& dtd, vector<double>& u, mesh& carcass, direct_task_data& dtd_synth);

#endif
