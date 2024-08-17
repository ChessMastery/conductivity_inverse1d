#include <iostream>
#include <fstream>
#include <algorithm>

#include "direct_task.h"
#include "dt_matrices.h"
#include "dt_functions.h"
#include "solver.h"

// lambda = 1 / (mu * mu0), mu = 1 for now (a dimensionless value)
// mu0 (vacuum magnetic permeability) = 4.0 * Pi * 1e-7 [ N * (A ^ -2) ]
extern const double lambda = 1.0 / (4.0 * 3.14159265358979323846 * 1e-7);

// default sigma (condictivity) = 0.1 [ S / m ] (siemens per meter)
extern const double sigma = 0.1;
// air conductivity is for real different from 0, but we neglect this fact

int time_index;
vector<point> nodes;
vector<elem> elems, elems_prim;
vector<vector<double>> solutions_vec_t_prim, solutions_vec_t_anom, prim_field;

extern vector<double> di_A, ggl_A, b, q;
extern vector<int> ia, ja;

double exp_fast(double x, int n) // fast x^n computation
{
   double res = 1.0;
   while (n > 0)
   {
      if (n % 2)
         res *= x;
      x *= x;
      n /= 2;
   }
   return res;
}

void build_fem_grid(direct_task_data& dtd_struct)
{
   int n = dtd_struct.r_coords.size() * dtd_struct.z_coords.size();
   int rn = dtd_struct.r_coords.size(), zn = dtd_struct.z_coords.size();
   nodes.clear();
   for (int i = 0; i < zn; i++)
      for (int j = 0; j < rn; j++)
         nodes.push_back(point(dtd_struct.r_coords[j], dtd_struct.z_coords[i]));
   nodes.shrink_to_fit();
   elems.clear();
   for (int i = 0; i < zn - 1; i++)
   {
      for (int j = 0; j < rn - 1; j++)
      {
         double sigma_el = 0.0;
         elem el_tmp(dtd_struct.r_coords[j], dtd_struct.r_coords[j + 1], dtd_struct.z_coords[i], dtd_struct.z_coords[i + 1],
            i * rn + j, i * rn + j + 1, (i + 1) * rn + j, (i + 1) * rn + j + 1, 0.0);
         // setting layers conductivity (sigma)
         if ((el_tmp.z1 + el_tmp.z0) / 2.0 < 0.0)
            el_tmp.sigma_el = sigma;
         double depth = 0.0;
         for (int k = 0; k < dtd_struct.layers_height.size(); k++)
         {
            depth += dtd_struct.layers_height[k];
            if ((el_tmp.z1 + el_tmp.z0) / 2.0 <= -depth + dtd_struct.layers_height[k] &&
               (el_tmp.z1 + el_tmp.z0) / 2.0 >= -depth)
               el_tmp.sigma_el = dtd_struct.layers_sigma[k];
         }

         elems.push_back(el_tmp);
      }
   }
   elems.shrink_to_fit();
}

void build_prim_fem_grid(direct_task_data& dtd_struct)
{
   int n = dtd_struct.r_coords.size() * dtd_struct.z_coords.size();
   int rn = dtd_struct.r_coords.size(), zn = dtd_struct.z_coords.size();
   nodes.clear();
   for (int i = 0; i < zn; i++)
      for (int j = 0; j < rn; j++)
         nodes.push_back(point(dtd_struct.r_coords[j], dtd_struct.z_coords[i]));
   nodes.shrink_to_fit();
   elems.clear();
   for (int i = 0; i < zn - 1; i++)
   {
      for (int j = 0; j < rn - 1; j++)
      {
         double sigma_el = 0.0;
         elem el_tmp(dtd_struct.r_coords[j], dtd_struct.r_coords[j + 1], dtd_struct.z_coords[i], dtd_struct.z_coords[i + 1],
            i * rn + j, i * rn + j + 1, (i + 1) * rn + j, (i + 1) * rn + j + 1, 0.0);
         // setting layers conductivity (sigma)
         if ((el_tmp.z1 + el_tmp.z0) / 2.0 < 0.0)
            el_tmp.sigma_el = sigma;

         elems.push_back(el_tmp);
      }
   }
   elems.shrink_to_fit();
}

void get_prim_mesh(direct_task_data& dtd) // todo: write a function forming x and z coords (code duplication)
{
   dtd.r_coords.clear();
   dtd.z_coords.clear();
   std::ifstream mesh_input;
   mesh_input.open("dir_input/prim_mesh.txt");
   if (!mesh_input.is_open())
   {
      std::cerr << "could not open file dir_input/prim_mesh.txt" << std::endl;
      throw std::runtime_error("could not open file dir_input/prim_mesh.txt");
   }
   int r_seg_num; // the number of r-coord discharge segments
   mesh_input >> r_seg_num;
   for (int i = 0; i < r_seg_num; i++)
   {
      double beg, end, coef; // discharge coefficient must be >= 1.0
      int inner_points_num, mode;
      mesh_input >> beg >> end >> coef >> inner_points_num >> mode;
      double step = (end - beg) * (1.0 - coef) / (1.0 - exp_fast(coef, inner_points_num + 1));
      dtd.r_coords.push_back(beg);
      if (mode == 1)
      {
         double coord = beg + step;
         while (coord < end)
         {
            dtd.r_coords.push_back(coord);
            step *= coef;
            coord += step;
         }
      }
      else if (mode == -1)
      {
         double coord = end - step;
         while (coord > beg)
         {
            dtd.r_coords.push_back(coord);
            step *= coef;
            coord -= step;
         }
      }
      dtd.r_coords.push_back(end);
   }
   int z_seg_num;
   mesh_input >> z_seg_num;
   for (int i = 0; i < z_seg_num; i++)
   {
      double beg, end, coef; // discharge coefficient must be >= 1.0
      int inner_points_num, mode;
      mesh_input >> beg >> end >> coef >> inner_points_num >> mode;
      double step = (end - beg) * (1.0 - coef) / (1.0 - exp_fast(coef, inner_points_num + 1));
      dtd.z_coords.push_back(beg);
      if (mode == 1)
      {
         double coord = beg + step;
         while (coord < end)
         {
            dtd.z_coords.push_back(coord);
            step *= coef;
            coord += step;
         }
      }
      else if (mode == -1)
      {
         double coord = end - step;
         while (coord > beg)
         {
            dtd.z_coords.push_back(coord);
            step *= coef;
            coord -= step;
         }
      }
      dtd.z_coords.push_back(end);
   }
   mesh_input.close();
   dtd.r_coords.push_back(dtd.src_r);
   dtd.z_coords.push_back(dtd.src_z);

   // adding layers boarders into z-coord vector
   double depth = 0.0;
   dtd.z_coords.push_back(-depth);
   for (int i = 0; i < dtd.layers_height.size(); i++)
   {
      depth += dtd.layers_height[i];
      dtd.z_coords.push_back(-depth);
   }

   // removing coord duplicates
   sort_and_remove_duplicates(dtd.r_coords);
   sort_and_remove_duplicates(dtd.z_coords);
   sort_and_remove_duplicates(dtd.vec_t);

   for (int i = 0; i < dtd.layers_sigma.size(); i++)
      dtd.layers_sigma[i] = sigma;
}

void calculate_prim(direct_task_data& dtd_struct, direct_task_data& dtd_prim)
{
   dtd_prim = dtd_struct;
   get_prim_mesh(dtd_prim);

   prim_field.clear();
   build_prim_fem_grid(dtd_prim);
   build_portrait();
   build_G();
   build_M();
   build_Mrr();

   time_index = 0; // computing q0 - the stationary problem solution
   build_A_t0();
   build_b_t0(dtd_prim);
   bc_1(dtd_prim);
   los_sparse_sst(ggl_A, di_A, ia, ja, q, b, nodes.size(), dtd_prim);
   prim_field.push_back(q);

   time_index = 1; // computing q1
   build_A_t1(dtd_prim);
   build_b_t1(dtd_prim);
   bc_1(dtd_prim);
   los_sparse_sst(ggl_A, di_A, ia, ja, q, b, nodes.size(), dtd_prim);
   prim_field.push_back(q);

   for (int i = 2; i < dtd_prim.vec_t.size(); i++)
   {
      time_index++;
      build_A(dtd_prim);
      build_b(dtd_prim);

      bc_1(dtd_prim);
      los_sparse_sst(ggl_A, di_A, ia, ja, q, b, nodes.size(), dtd_prim);
      prim_field.push_back(q);
   }
   elems_prim = elems;
}

void build_vec_t_prim(direct_task_data& dtd_struct, direct_task_data& dtd_prim)
{
   solutions_vec_t_prim.clear();
   solutions_vec_t_prim.resize(dtd_struct.vec_t.size());
   for (int k = 0; k < dtd_struct.vec_t.size(); k++)
   {
      //for (int i = 0; i < nodes.size(); i++)
      //   solutions_vec_t_prim[k].push_back(u_in_point(nodes[i].r, nodes[i].z, elems_prim, prim_field, k));
      for (int i = 0; i < dtd_struct.z_coords.size(); i++)
      {
         for (int j = 0; j < dtd_struct.r_coords.size(); j++)
         {
            solutions_vec_t_prim[k].push_back(u_in_point(dtd_struct.r_coords[j], dtd_struct.z_coords[i],
               dtd_prim.r_coords, dtd_prim.z_coords, elems_prim, prim_field, k));
         }
      }
   }
}

void solve_direct(direct_task_data& dtd_struct, direct_task_data& dtd_prim)
{
   // forms the vector solutions_vec_t_prim by getting primary field in the nodes of anomal field problem grid

   build_vec_t_prim(dtd_struct, dtd_prim);

   solutions_vec_t_anom.clear();
   build_fem_grid(dtd_struct);
   build_portrait();
   build_G();
   build_M();
   build_Mrr();
   build_M_anom(dtd_struct);

   q.clear();
   q.resize(dtd_struct.r_coords.size() * dtd_struct.z_coords.size(), 0.0);
   solutions_vec_t_anom.push_back(q);

   time_index = 1; // computing q1
   build_A_t1(dtd_struct);
   build_b_t1_anom(dtd_struct);
   bc_1(dtd_struct);
   los_sparse_sst(ggl_A, di_A, ia, ja, q, b, nodes.size(), dtd_struct);
   solutions_vec_t_anom.push_back(q);

   for (int i = 2; i < dtd_struct.vec_t.size(); i++)
   {
      time_index++;
      build_A(dtd_struct);
      build_b_anom(dtd_struct);

      bc_1(dtd_struct);
      los_sparse_sst(ggl_A, di_A, ia, ja, q, b, nodes.size(), dtd_struct);
      solutions_vec_t_anom.push_back(q);
   }
}

void get_direct_task_data(direct_task_data& dtd_struct, mesh& carcass) // quite an awkward multitask function, should be splitted
{
   std::ifstream mesh_input;
   mesh_input.open("dir_input/mesh.txt");
   if (!mesh_input.is_open())
   {
      std::cerr << "could not open file dir_input/mesh.txt" << std::endl;
      throw std::runtime_error("could not open file dir_input/mesh.txt");
   }
   int r_seg_num; // the number of r-coord discharge segments
   mesh_input >> r_seg_num;
   for (int i = 0; i < r_seg_num; i++)
   {
      double beg, end, coef; // discharge coefficient must be >= 1.0
      int inner_points_num, mode;
      mesh_input >> beg >> end >> coef >> inner_points_num >> mode;
      double step = (end - beg) * (1.0 - coef) / (1.0 - exp_fast(coef, inner_points_num + 1));
      dtd_struct.r_coords.push_back(beg);
      if (mode == 1)
      {
         double coord = beg + step;
         while (coord < end)
         {
            dtd_struct.r_coords.push_back(coord);
            step *= coef;
            coord += step;
         }
      }
      else if (mode == -1)
      {
         double coord = end - step;
         while (coord > beg)
         {
            dtd_struct.r_coords.push_back(coord);
            step *= coef;
            coord -= step;
         }
      }
      dtd_struct.r_coords.push_back(end);
   }
   int z_seg_num;
   mesh_input >> z_seg_num;
   for (int i = 0; i < z_seg_num; i++)
   {
      double beg, end, coef; // discharge coefficient must be >= 1.0
      int inner_points_num, mode;
      mesh_input >> beg >> end >> coef >> inner_points_num >> mode;
      double step = (end - beg) * (1.0 - coef) / (1.0 - exp_fast(coef, inner_points_num + 1));
      dtd_struct.z_coords.push_back(beg);
      if (mode == 1)
      {
         double coord = beg + step;
         while (coord < end)
         {
            dtd_struct.z_coords.push_back(coord);
            step *= coef;
            coord += step;
         }
      }
      else if (mode == -1)
      {
         double coord = end - step;
         while (coord > beg)
         {
            dtd_struct.z_coords.push_back(coord);
            step *= coef;
            coord -= step;
         }
      }
      dtd_struct.z_coords.push_back(end);
   }
   int time_seg_num;
   mesh_input >> time_seg_num;
   for (int i = 0; i < time_seg_num; i++)
   {
      double beg, end, coef; // discharge coefficient must be >= 1.0
      int inner_points_num, mode;
      mesh_input >> beg >> end >> coef >> inner_points_num >> mode;
      double step = (end - beg) * (1.0 - coef) / (1.0 - exp_fast(coef, inner_points_num + 1));
      dtd_struct.vec_t.push_back(beg);
      if (mode == 1)
      {
         double coord = beg + step;
         while (coord < end)
         {
            dtd_struct.vec_t.push_back(coord);
            step *= coef;
            coord += step;
         }
      }
      else if (mode == -1)
      {
         double coord = end - step;
         while (coord > beg)
         {
            dtd_struct.vec_t.push_back(coord);
            step *= coef;
            coord -= step;
         }
      }
      dtd_struct.vec_t.push_back(end);
   }
   mesh_input.close();

   std::ifstream src_input;
   src_input.open("dir_input/src.txt");
   if (!src_input.is_open())
   {
      std::cerr << "could not open file dir_input/src.txt" << std::endl;
      throw std::runtime_error("could not open file dir_input/src.txt");
   }
   src_input >> dtd_struct.src_r >> dtd_struct.src_z;
   src_input.close();
   dtd_struct.r_coords.push_back(dtd_struct.src_r);
   dtd_struct.z_coords.push_back(dtd_struct.src_z);

   carcass.r_coords = dtd_struct.r_coords;
   carcass.z_coords = dtd_struct.z_coords;

   std::ifstream layers_input;
   layers_input.open("dir_input/layers.txt");
   if (!layers_input.is_open())
   {
      std::cerr << "could not open file dir_input/layers.txt" << std::endl;
      throw std::runtime_error("could not open file dir_input/layers.txt");
   }
   int layers_num;
   layers_input >> layers_num;
   dtd_struct.layers_sigma.resize(layers_num);
   dtd_struct.layers_height.resize(layers_num);
   for (int i = 0; i < layers_num; i++)
      layers_input >> dtd_struct.layers_sigma[i] >> dtd_struct.layers_height[i];
   layers_input.close();
   // adding layers' boarders into z-coord vector
   double depth = 0.0;
   dtd_struct.z_coords.push_back(-depth);
   for (int i = 0; i < layers_num; i++)
   {
      depth += dtd_struct.layers_height[i];
      dtd_struct.z_coords.push_back(-depth);
   }

   // removing coord duplicates
   sort_and_remove_duplicates(dtd_struct.r_coords);
   sort_and_remove_duplicates(dtd_struct.z_coords);
   sort_and_remove_duplicates(dtd_struct.vec_t);
   sort_and_remove_duplicates(carcass.r_coords);
   sort_and_remove_duplicates(carcass.z_coords);
}

void sort_and_remove_duplicates(vector<double>& v)
{
   sort(v.begin(), v.end());
   const double dbl_eps = 1e-16;
   const double multiplier = 1e3; // determines the relative precision of the minimal possible step
   int pos = 0, gap = 0;
   for (int i = 1; i < v.size(); i++, pos++)
   {
      if (abs(v[i] - v[pos]) > multiplier * dbl_eps * (abs(v[i]) + abs(v[pos])))
      {
         if (gap)
            v[i - gap] = v[i];
      }
      else
      {
         gap++;
      }
   }
   v.resize(v.size() - gap);
   v.shrink_to_fit();
}

void calculate_eds(direct_task_data& dtd_struct, direct_task_data& dtd_prim, vector<double>& eds_vec)
{
   double R_acceptor = 3.0;
   const double eds_const = -2.0 * 3.14159265358979323846 * R_acceptor;

   //cout << "u(t=0): " << u_in_point(R_acceptor, dtd_struct.src_z, 0) << endl;

   //ofstream out_eds_t;
   //out_eds_t.open("eds_t.txt");
   //out_eds_t.precision(16);

   // output of the emf curve approximated with 3-dots quadratic lagrange spline on each step and taken in the middle point
   eds_vec.clear();
   for (int i = 2; i < dtd_struct.vec_t.size(); i++)
   {
      double dt1 = dtd_struct.vec_t[i - 1] - dtd_struct.vec_t[i - 2];
      double dt0 = dtd_struct.vec_t[i] - dtd_struct.vec_t[i - 1];
      double dt = dtd_struct.vec_t[i] - dtd_struct.vec_t[i - 2];

      double eds_prim = 0.0, eds_anom = 0.0;
      double _coef1 = dt1 / (dt0 * dt);
      double _coef2 = (dt0 - dt1) / (dt1 * dt0);
      double _coef3 = -dt0 / (dt1 * dt);
      eds_prim = eds_const * (_coef1 * u_in_point(R_acceptor, dtd_struct.src_z, dtd_prim.r_coords, dtd_prim.z_coords, elems_prim, prim_field, i) +
         _coef2 * u_in_point(R_acceptor, dtd_struct.src_z, dtd_prim.r_coords, dtd_prim.z_coords, elems_prim, prim_field, i - 1) +
         _coef3 * u_in_point(R_acceptor, dtd_struct.src_z, dtd_prim.r_coords, dtd_prim.z_coords, elems_prim, prim_field, i - 2));
      eds_anom = eds_const * (_coef1 * u_in_point(R_acceptor, dtd_struct.src_z, dtd_struct.r_coords, dtd_struct.z_coords, elems, solutions_vec_t_anom, i) +
         _coef2 * u_in_point(R_acceptor, dtd_struct.src_z, dtd_struct.r_coords, dtd_struct.z_coords, elems, solutions_vec_t_anom, i - 1) +
         _coef3 * u_in_point(R_acceptor, dtd_struct.src_z, dtd_struct.r_coords, dtd_struct.z_coords, elems, solutions_vec_t_anom, i - 2));
      eds_vec.push_back(eds_prim + eds_anom);

      //out_eds_t << eds_prim << " " << eds_anom << " " << eds_prim + eds_anom << endl;
   }
   //out_eds_t.close();
}