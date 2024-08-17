#ifndef DIRECT_TASK_H_SENTRY
#define DIRECT_TASK_H_SENTRY

#include <vector>
using std::vector;

struct point
{
   double r, z;
   point() : r(0.0), z(0.0) {}
   point(double p_r, double p_z) : r(p_r), z(p_z) {}
};

struct elem // rectangle element
{
   double r0, r1, z0, z1;
   int n1, n2, n3, n4; // global nodes numbers: lower left, lower right, upper left, upper right
   double sigma_el;
   elem() : r0(0.0), r1(0.0), z0(0.0), z1(0.0), n1(0), n2(0), n3(0), n4(0), sigma_el(0.0) {}
   elem(double r0_, double r1_, double z0_, double z1_, int n1_, int n2_, int n3_, int n4_, double sigma_el_)
      : r0(r0_), r1(r1_), z0(z0_), z1(z1_), n1(n1_), n2(n2_), n3(n3_), n4(n4_), sigma_el(sigma_el_) {}
};

struct direct_task_data
{
   vector<double> r_coords, z_coords;
   vector<double> vec_t; // the vector of the time layers

   double src_r, src_z;

   vector<double> layers_sigma; // sigma from top to the bottom
   vector<double> layers_height; // height of the layers from the top to the bottom one
};

struct mesh
{
   vector<double> r_coords, z_coords;
};

void get_direct_task_data(direct_task_data& dtd_struct, mesh& carcass);
void solve_direct(direct_task_data& dtd_struct, direct_task_data& dtd_prim);

void sort_and_remove_duplicates(vector<double>& v);
void calculate_prim(direct_task_data& dtd_struct, direct_task_data& dtd_prim);
void build_M_anom(direct_task_data& dtd_struct);
void calculate_eds(direct_task_data& dtd_struct, direct_task_data& dtd_prim, vector<double>& eds_vec);

#endif
