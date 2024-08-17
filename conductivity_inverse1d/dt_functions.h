#ifndef DT_FUNCTIONS_H_SENTRY
#define DT_FUNCTIONS_H_SENTRY

#include "direct_task.h"

double f_rzt_right_side(double r, double z, double t);

double ksi1_loc(double _r, double _z, int _elem_num, vector<elem>& elements);
double ksi2_loc(double _r, double _z, int _elem_num, vector<elem>& elements);
double ksi3_loc(double _r, double _z, int _elem_num, vector<elem>& elements);
double ksi4_loc(double _r, double _z, int _elem_num, vector<elem>& elements);

void bc_1(direct_task_data& dtd_struct);

//void output_for_python_plots(direct_task_data& dtd_struct);

double u_in_point(double r, double z, vector<double>& r_coords, vector<double>& z_coords,
   vector<elem>& elements, vector<vector<double>>& sol, int q_ind);

#endif

