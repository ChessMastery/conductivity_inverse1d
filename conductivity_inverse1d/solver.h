#ifndef SOLVER_H_SENTRY
#define SOLVER_H_SENTRY

#include "direct_task.h"

void los_sparse_sst(vector<double>& gg, vector<double>& di, vector<int>& ig, vector<int>& jg,
   vector<double>& x, vector<double>& b, int n, direct_task_data& dtd_struct);

void gauss_v2(vector<vector<double>>& a, vector<double>& x, vector<double>& y, int n, double eps);

#endif
