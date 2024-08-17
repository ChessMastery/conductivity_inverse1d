#include "dt_functions.h"

#include <iostream>
#include <fstream>
#include <string>
using namespace std;

extern vector<elem> elems, elems_prim;
extern vector<double> di_A;
extern vector<double> b;
extern int time_index;
extern vector<vector<double>> solutions_vec_t_prim, solutions_vec_t_anom;

double f_rzt_right_side(double r, double z, double t)
{
   double res = 0.0;
   res = 0.0;

   return res;
}

double bc1_function(double r, double z, double t)
{
   double res = 0.0;
   res = 0.0;

   return res;
}

double ksi1_loc(double _r, double _z, int _elem_num, vector<elem>& elements)
{
   double res = 0.0;
   res = (elements[_elem_num].r1 - _r) / (elements[_elem_num].r1 - elements[_elem_num].r0) *
      (elements[_elem_num].z1 - _z) / (elements[_elem_num].z1 - elements[_elem_num].z0);
   return res;
}
double ksi2_loc(double _r, double _z, int _elem_num, vector<elem>& elements)
{
   double res = 0.0;
   res = (_r - elements[_elem_num].r0) / (elements[_elem_num].r1 - elements[_elem_num].r0) *
      (elements[_elem_num].z1 - _z) / (elements[_elem_num].z1 - elements[_elem_num].z0);
   return res;
}
double ksi3_loc(double _r, double _z, int _elem_num, vector<elem>& elements)
{
   double res = 0.0;
   res = (elements[_elem_num].r1 - _r) / (elements[_elem_num].r1 - elements[_elem_num].r0) *
      (_z - elements[_elem_num].z0) / (elements[_elem_num].z1 - elements[_elem_num].z0);
   return res;
}
double ksi4_loc(double _r, double _z, int _elem_num, vector<elem>& elements)
{
   double res = 0.0;
   res = (_r - elements[_elem_num].r0) / (elements[_elem_num].r1 - elements[_elem_num].r0) *
      (_z - elements[_elem_num].z0) / (elements[_elem_num].z1 - elements[_elem_num].z0);
   return res;
}

void bc_1(direct_task_data& dtd_struct)
{
   int n = dtd_struct.r_coords.size() * dtd_struct.z_coords.size();
   int nr = dtd_struct.r_coords.size(), nz = dtd_struct.z_coords.size();

   double C = 1e30;

   // lower bound (together with the left and right node)
   for (int i = 0; i < nr - 1; i++)
   {
      di_A[i] = C;
      b[i] = C * bc1_function(elems[i].r0, elems[i].z0, dtd_struct.vec_t[time_index]);
   }
   di_A[nr - 1] = C;
   b[nr - 1] = C * bc1_function(elems[nr - 2].r1, elems[nr - 2].z0, dtd_struct.vec_t[time_index]);

   ////left bound (without the lower and upper node)
   //for (int i = 1; i < nodes_z - 1; i++)
   //{
   //   di_A[i * nodes_r] = C;
   //   b[nodes_r * i] = C * test_func(elements[(nodes_r - 1) * i].r0, elements[(nodes_r - 1) * i].z0, vec_t[time_node_num]);
   //}

   // right bound (without the lower and upper node)
   for (int i = 2; i < nz; i++)
   {
      di_A[nr * i - 1] = C;
      b[nr * i - 1] = C * bc1_function(elems[(nr - 1) * i - 1].r1, elems[(nr - 1) * i - 1].z0, dtd_struct.vec_t[time_index]);
   }

   // upper bound (with the left and right node)
   for (int i = n - nr; i < n; i++)
      di_A[i] = C;

   for (int i = 0; i < nr - 1; i++)
      b[n - nr + i] = C * bc1_function(elems[elems.size() - nr + 1 + i].r0,
         elems[elems.size() - nr + 1 + i].z1, dtd_struct.vec_t[time_index]);
   b[n - 1] = C * bc1_function(elems[elems.size() - 1].r1, elems[elems.size() - 1].z1, dtd_struct.vec_t[time_index]);
}

void output_for_python_plots(direct_task_data& dtd_struct, vector<vector<double>>& solutions_vec_t) // direct problem output
{
   ofstream output_solution_and_coords;
   for (int i = 0; i < solutions_vec_t.size(); i++)
   {
      string path = "output/q" + to_string(i) + ".txt";
      output_solution_and_coords.open(path);
      //out << solutions_vec_t.size() << endl;
      for (int j = 0; j < solutions_vec_t[i].size(); j++)
         output_solution_and_coords << solutions_vec_t[i][j] << endl;
      output_solution_and_coords.close();
   }
   output_solution_and_coords.open("output/r_coords.txt");
   //out << r_nodes_vec.size() << endl;
   for (int i = 0; i < dtd_struct.r_coords.size(); i++)
      output_solution_and_coords << dtd_struct.r_coords[i] << endl;
   output_solution_and_coords.close();
   output_solution_and_coords.open("output/z_coords.txt");
   //out << z_nodes_vec.size() << endl;
   for (int i = 0; i < dtd_struct.z_coords.size(); i++)
      output_solution_and_coords << dtd_struct.z_coords[i] << endl;
   output_solution_and_coords.close();
}

double u_in_point(double r, double z, vector<double>& r_coords, vector<double>& z_coords,
   vector<elem>& elements, vector<vector<double>>& sol, int q_ind) // function providing solution with the use of numeration
{
   int elem_index = 0;
   int zi, ri;
   int lb = 0, rb = z_coords.size() - 1;
   while (rb - lb > 1)
   {
      int mid = lb + (rb - lb) / 2;
      if (z >= z_coords[mid])
         lb = mid;
      else
         rb = mid;
   }
   zi = lb;
   lb = 0, rb = r_coords.size() - 1;
   while (rb - lb > 1)
   {
      int mid = lb + (rb - lb) / 2;
      if (r >= r_coords[mid])
         lb = mid;
      else
         rb = mid;
   }
   ri = lb;

   elem_index = zi * (r_coords.size() - 1) + ri;

   double res = 0.0;
   res = sol[q_ind][elements[elem_index].n1] * ksi1_loc(r, z, elem_index, elements) +
      sol[q_ind][elements[elem_index].n2] * ksi2_loc(r, z, elem_index, elements) +
      sol[q_ind][elements[elem_index].n3] * ksi3_loc(r, z, elem_index, elements) +
      sol[q_ind][elements[elem_index].n4] * ksi4_loc(r, z, elem_index, elements);

   return res;
}