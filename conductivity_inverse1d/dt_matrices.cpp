#include "dt_matrices.h"
#include "dt_functions.h"
#include <iostream>
#include <vector>
#include <map>
#include <set>
using std::set;
using std::map;
using std::pair;

extern const double lambda;
extern const double sigma;

extern vector<elem> elems;
extern vector<point> nodes;

extern int time_index;
extern vector<vector<double>> solutions_vec_t_prim, solutions_vec_t_anom, prim_field;

vector<int> ia, ja; // the portrait of matrices
vector<set<int>> L; // adjacency list

// stiffness matrix
map<pair<int, int>, double> G; // values
vector<double> di_G, ggl_G;

// mass matrix
map<pair<int, int>, double> M;
vector<double> di_M, ggl_M;

map<pair<int, int>, double> M_anom;
vector<double> di_M_anom, ggl_M_anom;

// mass matrix with 1/r^2 multiplier
map<pair<int, int>, double> Mrr;
vector<double> di_Mrr, ggl_Mrr;

vector<double> di_A, ggl_A; // SLAE Aq = b matrix
vector<double> b, q; // SLAE Aq = b vectors

void add_to_adjlist(int ind1, int ind2)
{
   if (ind1 < ind2) std::swap(ind1, ind2);
   L[ind1].insert(ind2);
}

void build_portrait()
{
   G.clear(); M.clear(); Mrr.clear(); M_anom.clear();
   ia.clear(); ja.clear();
   L.clear();
   L.resize(nodes.size());
   for (int i = 0; i < elems.size(); i++)
   {
      add_to_adjlist(elems[i].n1, elems[i].n2);
      add_to_adjlist(elems[i].n1, elems[i].n3);
      add_to_adjlist(elems[i].n1, elems[i].n4);

      add_to_adjlist(elems[i].n2, elems[i].n3);
      add_to_adjlist(elems[i].n2, elems[i].n4);

      add_to_adjlist(elems[i].n3, elems[i].n4);
   }
   ia.push_back(0);
   int ia_cnt = 0;
   for (int i = 0; i < L.size(); i++)
   {
      for (auto col : L[i])
      {
         ja.push_back(col);
         ia_cnt++;
      }
      ia.push_back(ia_cnt);
   }
   ia.shrink_to_fit();
   ja.shrink_to_fit();
}

void add_to_sparse_format(int ind1, int ind2, double value, vector<set<int>>& L, map<pair<int, int>, double>& M)
{
   if (ind1 < ind2) std::swap(ind1, ind2);
   if (M.count({ ind1, ind2 }))
      M[{ind1, ind2}] += value;
   else
      M[{ind1, ind2}] = value;
}

void build_G()
{
   ggl_G.clear();
   di_G.resize(nodes.size());
   for (int i = 0; i < di_G.size(); i++) di_G[i] = 0.0;
   double h_r = 0, h_z = 0, r_0 = 0, r_1 = 0, z_0 = 0, z_1 = 0;
   double coef_1_tmp = 0, coef_2_tmp = 0;

   vector<vector<double>> G_loc(4, vector<double>(4, 0.0));
   for (int i = 0; i < elems.size(); i++)
   {
      // local stiffness matrix G_loc computation
      r_0 = elems[i].r0;
      r_1 = elems[i].r1;
      z_0 = elems[i].z0;
      z_1 = elems[i].z1;
      h_r = r_1 - r_0;
      h_z = z_1 - z_0;

      coef_1_tmp = lambda * h_z * (r_1 * r_1 - r_0 * r_0) / (h_r * h_r * 2.0);
      coef_2_tmp = lambda * 1.0 / (h_z * 12.0);

      G_loc[0][0] = coef_1_tmp / 3.0 + coef_2_tmp * (r_1 + 3.0 * r_0) * h_r;
      G_loc[0][1] = -coef_1_tmp / 3.0 + coef_2_tmp * (r_1 * r_1 - r_0 * r_0);
      G_loc[0][2] = coef_1_tmp / 6.0 - coef_2_tmp * (r_1 + 3.0 * r_0) * h_r;
      G_loc[0][3] = -coef_1_tmp / 6.0 - coef_2_tmp * (r_1 * r_1 - r_0 * r_0);

      G_loc[1][1] = coef_1_tmp / 3.0 + coef_2_tmp * (3.0 * r_1 + r_0) * h_r;
      G_loc[1][2] = -coef_1_tmp / 6.0 - coef_2_tmp * (r_1 * r_1 - r_0 * r_0);
      G_loc[1][3] = coef_1_tmp / 6.0 - coef_2_tmp * (3.0 * r_1 + r_0) * h_r;

      G_loc[2][2] = coef_1_tmp / 3.0 + coef_2_tmp * (r_1 + 3.0 * r_0) * h_r;
      G_loc[2][3] = -coef_1_tmp / 3.0 + coef_2_tmp * (r_1 * r_1 - r_0 * r_0);

      G_loc[3][3] = coef_1_tmp / 3.0 + coef_2_tmp * (3.0 * r_1 + r_0) * h_r;

      // including G_loc into the global one G
      di_G[elems[i].n1] += G_loc[0][0];
      add_to_sparse_format(elems[i].n1, elems[i].n2, G_loc[0][1], L, G);
      add_to_sparse_format(elems[i].n1, elems[i].n3, G_loc[0][2], L, G);
      add_to_sparse_format(elems[i].n1, elems[i].n4, G_loc[0][3], L, G);

      di_G[elems[i].n2] += G_loc[1][1];
      add_to_sparse_format(elems[i].n2, elems[i].n3, G_loc[1][2], L, G);
      add_to_sparse_format(elems[i].n2, elems[i].n4, G_loc[1][3], L, G);

      di_G[elems[i].n3] += G_loc[2][2];
      add_to_sparse_format(elems[i].n3, elems[i].n4, G_loc[2][3], L, G);

      di_G[elems[i].n4] += G_loc[3][3];
   }

   // form the sparse format array
   for (int i = 0; i < L.size(); i++)
      for (auto col : L[i])
         ggl_G.push_back(G[{i, col}]);
   ggl_G.shrink_to_fit();
}

void build_M()
{
   ggl_M.clear();
   di_M.resize(nodes.size());
   for (int i = 0; i < di_M.size(); i++) di_M[i] = 0.0;
   double h_r = 0, h_z = 0, r_0 = 0, r_1 = 0, z_0 = 0, z_1 = 0;
   double coef_tmp;

   vector<vector<double>> M_loc(4, vector<double>(4, 0.0));
   for (int i = 0; i < elems.size(); i++)
   {
      // local mass matrix M_loc computation
      r_0 = elems[i].r0;
      r_1 = elems[i].r1;
      z_0 = elems[i].z0;
      z_1 = elems[i].z1;
      h_r = r_1 - r_0;
      h_z = z_1 - z_0;

      coef_tmp = elems[i].sigma_el * h_z / 12.0;

      M_loc[0][0] = coef_tmp * h_r / 3.0 * (r_1 + 3.0 * r_0);
      M_loc[0][1] = coef_tmp * (r_1 * r_1 - r_0 * r_0) / 3.0;
      M_loc[0][2] = coef_tmp * h_r / 6.0 * (r_1 + 3.0 * r_0);
      M_loc[0][3] = coef_tmp * (r_1 * r_1 - r_0 * r_0) / 6.0;

      M_loc[1][1] = coef_tmp * h_r / 3.0 * (3.0 * r_1 + r_0);
      M_loc[1][2] = coef_tmp * (r_1 * r_1 - r_0 * r_0) / 6.0;
      M_loc[1][3] = coef_tmp * h_r / 6.0 * (3.0 * r_1 + r_0);

      M_loc[2][2] = coef_tmp * h_r / 3.0 * (r_1 + 3 * r_0);
      M_loc[2][3] = coef_tmp * (r_1 * r_1 - r_0 * r_0) / 3.0;

      M_loc[3][3] = coef_tmp * h_r / 3.0 * (3.0 * r_1 + r_0);

      // including M_loc into the global one M
      di_M[elems[i].n1] += M_loc[0][0];
      add_to_sparse_format(elems[i].n1, elems[i].n2, M_loc[0][1], L, M);
      add_to_sparse_format(elems[i].n1, elems[i].n3, M_loc[0][2], L, M);
      add_to_sparse_format(elems[i].n1, elems[i].n4, M_loc[0][3], L, M);

      di_M[elems[i].n2] += M_loc[1][1];
      add_to_sparse_format(elems[i].n2, elems[i].n3, M_loc[1][2], L, M);
      add_to_sparse_format(elems[i].n2, elems[i].n4, M_loc[1][3], L, M);

      di_M[elems[i].n3] += M_loc[2][2];
      add_to_sparse_format(elems[i].n3, elems[i].n4, M_loc[2][3], L, M);

      di_M[elems[i].n4] += M_loc[3][3];
   }

   // form the sparse format array
   for (int i = 0; i < L.size(); i++)
      for (auto col : L[i])
         ggl_M.push_back(M[{i, col}]);
   ggl_M.shrink_to_fit();
}

void build_Mrr()
{
   ggl_Mrr.clear();
   di_Mrr.resize(nodes.size());
   for (int i = 0; i < di_Mrr.size(); i++) di_Mrr[i] = 0.0;
   double h_r = 0, h_z = 0, r_0 = 0, r_1 = 0, z_0 = 0, z_1 = 0;
   double coef_tmp = 0.0;

   vector<vector<double>> Mrr_loc(4, vector<double>(4, 0.0));
   for (int i = 0; i < elems.size(); i++)
   {
      // local mass matrix M_loc with 1/r^2 multiplier computation
      r_0 = elems[i].r0;
      r_1 = elems[i].r1;
      z_0 = elems[i].z0;
      z_1 = elems[i].z1;
      h_r = r_1 - r_0;
      h_z = z_1 - z_0;

      coef_tmp = lambda * h_z / (3.0 * h_r * h_r);

      double r1r1 = r_1 * r_1 * log(r_1 / r_0) + 2.0 * r_1 * r_0 - (3.0 * r_1 * r_1 + r_0 * r_0) / 2.0;
      double r2r2 = r_0 * r_0 * log(r_1 / r_0) - 2.0 * r_1 * r_0 + (3.0 * r_0 * r_0 + r_1 * r_1) / 2.0;
      double r1r2 = (r_1 * r_1 - r_0 * r_0) / 2.0 + r_1 * r_0 * -log(r_1 / r_0);


      Mrr_loc[0][0] = coef_tmp * r1r1;
      Mrr_loc[0][1] = coef_tmp * r1r2;
      Mrr_loc[0][2] = coef_tmp / 2.0 * r1r1;
      Mrr_loc[0][3] = coef_tmp / 2.0 * r1r2;

      Mrr_loc[1][1] = coef_tmp * r2r2;
      Mrr_loc[1][2] = coef_tmp / 2.0 * r1r2;
      Mrr_loc[1][3] = coef_tmp / 2.0 * r2r2;

      Mrr_loc[2][2] = coef_tmp * r1r1;
      Mrr_loc[2][3] = coef_tmp * r1r2;

      Mrr_loc[3][3] = coef_tmp * r2r2;
      // including Mrr_loc into the global one Mrr
      di_Mrr[elems[i].n1] += Mrr_loc[0][0];
      add_to_sparse_format(elems[i].n1, elems[i].n2, Mrr_loc[0][1], L, Mrr);
      add_to_sparse_format(elems[i].n1, elems[i].n3, Mrr_loc[0][2], L, Mrr);
      add_to_sparse_format(elems[i].n1, elems[i].n4, Mrr_loc[0][3], L, Mrr);

      di_Mrr[elems[i].n2] += Mrr_loc[1][1];
      add_to_sparse_format(elems[i].n2, elems[i].n3, Mrr_loc[1][2], L, Mrr);
      add_to_sparse_format(elems[i].n2, elems[i].n4, Mrr_loc[1][3], L, Mrr);

      di_Mrr[elems[i].n3] += Mrr_loc[2][2];
      add_to_sparse_format(elems[i].n3, elems[i].n4, Mrr_loc[2][3], L, Mrr);

      di_Mrr[elems[i].n4] += Mrr_loc[3][3];
   }

   // form the sparse format array
   for (int i = 0; i < L.size(); i++)
      for (auto col : L[i])
         ggl_Mrr.push_back(Mrr[{i, col}]);
   ggl_Mrr.shrink_to_fit();
}

void build_M_anom(direct_task_data& dtd_struct)
{
   ggl_M_anom.clear();
   di_M_anom.resize(nodes.size());
   for (int i = 0; i < di_M_anom.size(); i++)
      di_M_anom[i] = 0.0;

   double h_r = 0, h_z = 0, r_0 = 0, r_1 = 0, z_0 = 0, z_1 = 0;
   double coef_tmp;


   vector<vector<double>> M_anom_loc(4, vector<double>(4, 0.0));
   //int el_sigm_cnt = 0;
   for (int i = 0; i < elems.size(); i++)
   {
      // mass matrix computation for the component -(sigma-sigma0)(du0/dt)
      r_0 = elems[i].r0;
      r_1 = elems[i].r1;
      z_0 = elems[i].z0;
      z_1 = elems[i].z1;
      h_r = r_1 - r_0;
      h_z = z_1 - z_0;

      double el_sigma = 0.0;

      double depth = 0.0;
      for (int k = 0; k < dtd_struct.layers_height.size() - 1; k++)
      {
         if ((elems[i].z1 + elems[i].z0) / 2.0 <= -depth &&
            (elems[i].z1 + elems[i].z0) / 2.0 >= -depth - dtd_struct.layers_height[k])
         {
            el_sigma = sigma - elems[i].sigma_el;
            //el_sigm_cnt++;
         }
         depth += dtd_struct.layers_height[k];
      }

      coef_tmp = el_sigma * h_z / 12.0;

      M_anom_loc[0][0] = coef_tmp * h_r / 3.0 * (r_1 + 3.0 * r_0);
      M_anom_loc[0][1] = coef_tmp * (r_1 * r_1 - r_0 * r_0) / 3.0;
      M_anom_loc[0][2] = coef_tmp * h_r / 6.0 * (r_1 + 3.0 * r_0);
      M_anom_loc[0][3] = coef_tmp * (r_1 * r_1 - r_0 * r_0) / 6.0;

      M_anom_loc[1][1] = coef_tmp * h_r / 3.0 * (3.0 * r_1 + r_0);
      M_anom_loc[1][2] = coef_tmp * (r_1 * r_1 - r_0 * r_0) / 6.0;
      M_anom_loc[1][3] = coef_tmp * h_r / 6.0 * (3.0 * r_1 + r_0);

      M_anom_loc[2][2] = coef_tmp * h_r / 3.0 * (r_1 + 3 * r_0);
      M_anom_loc[2][3] = coef_tmp * (r_1 * r_1 - r_0 * r_0) / 3.0;

      M_anom_loc[3][3] = coef_tmp * h_r / 3.0 * (3.0 * r_1 + r_0);

      di_M_anom[elems[i].n1] += M_anom_loc[0][0];
      add_to_sparse_format(elems[i].n1, elems[i].n2, M_anom_loc[0][1], L, M_anom);
      add_to_sparse_format(elems[i].n1, elems[i].n3, M_anom_loc[0][2], L, M_anom);
      add_to_sparse_format(elems[i].n1, elems[i].n4, M_anom_loc[0][3], L, M_anom);

      di_M_anom[elems[i].n2] += M_anom_loc[1][1];
      add_to_sparse_format(elems[i].n2, elems[i].n3, M_anom_loc[1][2], L, M_anom);
      add_to_sparse_format(elems[i].n2, elems[i].n4, M_anom_loc[1][3], L, M_anom);

      di_M_anom[elems[i].n3] += M_anom_loc[2][2];
      add_to_sparse_format(elems[i].n3, elems[i].n4, M_anom_loc[2][3], L, M_anom);

      di_M_anom[elems[i].n4] += M_anom_loc[3][3];
   }

   // form the sparse format array
   for (int i = 0; i < L.size(); i++)
      for (auto col : L[i])
         ggl_M_anom.push_back(M_anom[{i, col}]);
   ggl_M_anom.shrink_to_fit();
}

void build_A_t0()
{
   di_A.resize(di_G.size());
   ggl_A.resize(ggl_G.size());
   for (int i = 0; i < di_A.size(); i++)
      di_A[i] = di_G[i] + di_Mrr[i];
   for (int i = 0; i < ggl_A.size(); i++)
      ggl_A[i] = ggl_G[i] + ggl_Mrr[i];
}

void build_b_t0(direct_task_data& dtd_struct)
{
   int n = dtd_struct.r_coords.size() * dtd_struct.z_coords.size();
   b.resize(n);
   for (int i = 0; i < n; i++) b[i] = 0.0;

   vector<double> b_loc(4, 0.0);
   for (int i = 0; i < elems.size(); i++)
   {
      // no integration is required as rhs contribution is zero except source delta-function
      // b_loc[i] = (integral) test_func * jacobian * basis_function over all the domain
      //b_loc[0] = integrate_frz_ksi_2Drect_Gauss5(f_rzt_right_side, ksi1_loc, i, dtd_struct);
      //b_loc[1] = integrate_frz_ksi_2Drect_Gauss5(f_rzt_right_side, ksi2_loc, i, dtd_struct);
      //b_loc[2] = integrate_frz_ksi_2Drect_Gauss5(f_rzt_right_side, ksi3_loc, i, dtd_struct);
      //b_loc[3] = integrate_frz_ksi_2Drect_Gauss5(f_rzt_right_side, ksi4_loc, i, dtd_struct);

      b[elems[i].n1] += b_loc[0];
      b[elems[i].n2] += b_loc[1];
      b[elems[i].n3] += b_loc[2];
      b[elems[i].n4] += b_loc[3];
   }

   // including the concentrated source (as delta-function in one of the mesh nodes)
   int src_node_index = 0;
   for (int i = 0; i < elems.size(); i++)
   {
      if (elems[i].r0 <= dtd_struct.src_r && dtd_struct.src_r <= elems[i].r1
         && elems[i].z0 <= dtd_struct.src_z && dtd_struct.src_z <= elems[i].z1)
      {
         vector<double> dist(4, 0.0);
         dist[0] = hypot(dtd_struct.src_r - elems[i].r0, dtd_struct.src_z - elems[i].z0);
         dist[1] = hypot(dtd_struct.src_r - elems[i].r1, dtd_struct.src_z - elems[i].z0);
         dist[2] = hypot(dtd_struct.src_r - elems[i].r0, dtd_struct.src_z - elems[i].z1);
         dist[3] = hypot(dtd_struct.src_r - elems[i].r1, dtd_struct.src_z - elems[i].z1);
         int min_index = -1;
         double min_dist = (elems[i].r1 - elems[i].r0 + elems[i].z1 - elems[i].z0) * 2.0;
         for (int k = 0; k < dist.size(); k++)
            if (dist[k] < min_dist)
            {
               min_dist = dist[k];
               min_index = k;
            }
         if (min_index == 0) src_node_index = elems[i].n1;
         else if (min_index == 1) src_node_index = elems[i].n2;
         else if (min_index == 2) src_node_index = elems[i].n3;
         else if (min_index == 3) src_node_index = elems[i].n4;
         break;
      }
   }

   double src_power = 1.0;
   b[src_node_index] += src_power * dtd_struct.src_r;
   // src_r multiplier comes from delta-function integration over the domain
   // (integral) delta_function * jacobian
}

void build_A_t1(direct_task_data& dtd_struct)
{
   di_A.resize(di_G.size());
   ggl_A.resize(ggl_G.size());
   for (int i = 0; i < di_A.size(); i++)
      di_A[i] = di_G[i] + di_Mrr[i];
   for (int i = 0; i < ggl_A.size(); i++)
      ggl_A[i] = ggl_G[i] + ggl_Mrr[i];

   double dt0 = 0.0;
   dt0 = dtd_struct.vec_t[time_index] - dtd_struct.vec_t[time_index - 1];
   for (int i = 0; i < di_A.size(); i++)
      di_A[i] += di_M[i] / dt0;
   for (int i = 0; i < ggl_A.size(); i++)
      ggl_A[i] += ggl_M[i] / dt0;
}

void build_b_t1(direct_task_data& dtd_struct)
{
   int n = dtd_struct.r_coords.size() * dtd_struct.z_coords.size();
   b.resize(n);
   for (int i = 0; i < n; i++) b[i] = 0.0;

   vector<double> b_loc(4, 0.0);
   for (int i = 0; i < elems.size(); i++)
   {
      // computing of the local vector b_loc components 
      // b_loc[i] = (integral) test_func * jacobian * basis_function over all the domain
      //b_loc[0] = integrate_frz_ksi_2Drect_Gauss5(f_rzt_right_side, ksi1_loc, i, dtd_struct);
      //b_loc[1] = integrate_frz_ksi_2Drect_Gauss5(f_rzt_right_side, ksi2_loc, i, dtd_struct);
      //b_loc[2] = integrate_frz_ksi_2Drect_Gauss5(f_rzt_right_side, ksi3_loc, i, dtd_struct);
      //b_loc[3] = integrate_frz_ksi_2Drect_Gauss5(f_rzt_right_side, ksi4_loc, i, dtd_struct);

      b[elems[i].n1] += b_loc[0];
      b[elems[i].n2] += b_loc[1];
      b[elems[i].n3] += b_loc[2];
      b[elems[i].n4] += b_loc[3];
   }

   double dt0 = 0.0;
   dt0 = dtd_struct.vec_t[time_index] - dtd_struct.vec_t[time_index - 1];
   vector<double> Mq(b.size(), 0.0);
   for (int j = 0; j < Mq.size(); j++)
   {
      Mq[j] += di_M[j] * prim_field[time_index - 1][j];
      for (int k = ia[j]; k < ia[j + 1]; k++)
      {
         Mq[j] += ggl_M[k] * prim_field[time_index - 1][ja[k]];
         Mq[ja[k]] += ggl_M[k] * prim_field[time_index - 1][j];
      }
   }

   for (int i = 0; i < b.size(); i++)
      b[i] += Mq[i] / dt0;
}

void build_b_t1_anom(direct_task_data& dtd_struct)
{
   int n = dtd_struct.r_coords.size() * dtd_struct.z_coords.size();
   b.resize(n);
   for (int i = 0; i < n; i++) b[i] = 0.0;
   double dt0 = 0.0;
   dt0 = dtd_struct.vec_t[time_index] - dtd_struct.vec_t[time_index - 1];
   vector<double> Mq(n, 0.0);
   vector<double> Mq_sigm_addf_j(n, 0.0);
   vector<double> Mq_sigm_addf_j_minus_1(n, 0.0);
   for (int j = 0; j < n; j++)
   {
      Mq[j] += di_M[j] * solutions_vec_t_anom[time_index - 1][j];
      Mq_sigm_addf_j[j] += di_M_anom[j] * solutions_vec_t_prim[time_index][j];
      Mq_sigm_addf_j_minus_1[j] += di_M_anom[j] * solutions_vec_t_prim[time_index - 1][j];
      for (int k = ia[j]; k < ia[j + 1]; k++)
      {
         Mq[j] += ggl_M[k] * solutions_vec_t_anom[time_index - 1][ja[k]];
         Mq[ja[k]] += ggl_M[k] * solutions_vec_t_anom[time_index - 1][j];

         Mq_sigm_addf_j[j] += ggl_M_anom[k] * solutions_vec_t_prim[time_index][ja[k]];
         Mq_sigm_addf_j[ja[k]] += ggl_M_anom[k] * solutions_vec_t_prim[time_index][j];

         Mq_sigm_addf_j_minus_1[j] += ggl_M_anom[k] * solutions_vec_t_prim[time_index - 1][ja[k]];
         Mq_sigm_addf_j_minus_1[ja[k]] += ggl_M_anom[k] * solutions_vec_t_prim[time_index - 1][j];
      }
   }

   for (int i = 0; i < n; i++)
   {
      b[i] += Mq[i] / dt0;
      b[i] += Mq_sigm_addf_j[i] / dt0;
      b[i] -= Mq_sigm_addf_j_minus_1[i] / dt0;
   }
}

void build_A(direct_task_data& dtd_struct)
{
   di_A.resize(di_G.size());
   ggl_A.resize(ggl_G.size());
   for (int i = 0; i < di_A.size(); i++)
      di_A[i] = di_G[i] + di_Mrr[i];
   for (int i = 0; i < ja.size(); i++)
      ggl_A[i] = ggl_G[i] + ggl_Mrr[i];

   double dt = 0, dt0 = 0, dt1 = 0, temp = 0;
   dt0 = dtd_struct.vec_t[time_index] - dtd_struct.vec_t[time_index - 1];
   dt1 = dtd_struct.vec_t[time_index - 1] - dtd_struct.vec_t[time_index - 2];
   dt = dtd_struct.vec_t[time_index] - dtd_struct.vec_t[time_index - 2];

   temp = (dt + dt0) / (dt * dt0);
   for (int i = 0; i < di_A.size(); i++)
      di_A[i] += temp * di_M[i];
   for (int i = 0; i < ggl_A.size(); i++)
      ggl_A[i] += temp * ggl_M[i];
}

void build_b(direct_task_data& dtd_struct)
{
   int n = dtd_struct.r_coords.size() * dtd_struct.z_coords.size();
   b.resize(n);
   for (int i = 0; i < n; i++) b[i] = 0.0;

   vector<double> b_loc(4, 0.0);
   for (int i = 0; i < elems.size(); i++)
   {
      // computing of the local vector b_loc components 
      // b_loc[i] = (integral) test_func * jacobian * basis_function over all the domain
      //b_loc[0] = integrate_frz_ksi_2Drect_Gauss5(f_rzt_right_side, ksi1_loc, i, dtd_struct);
      //b_loc[1] = integrate_frz_ksi_2Drect_Gauss5(f_rzt_right_side, ksi2_loc, i, dtd_struct);
      //b_loc[2] = integrate_frz_ksi_2Drect_Gauss5(f_rzt_right_side, ksi3_loc, i, dtd_struct);
      //b_loc[3] = integrate_frz_ksi_2Drect_Gauss5(f_rzt_right_side, ksi4_loc, i, dtd_struct);

      b[elems[i].n1] += b_loc[0];
      b[elems[i].n2] += b_loc[1];
      b[elems[i].n3] += b_loc[2];
      b[elems[i].n4] += b_loc[3];
   }

   double dt = 0.0, dt0 = 0.0, dt1 = 0.0, temp1 = 0.0, temp2 = 0.0;
   dt0 = dtd_struct.vec_t[time_index] - dtd_struct.vec_t[time_index - 1];
   dt1 = dtd_struct.vec_t[time_index - 1] - dtd_struct.vec_t[time_index - 2];
   dt = dtd_struct.vec_t[time_index] - dtd_struct.vec_t[time_index - 2];

   temp1 = dt / (dt1 * dt0);
   temp2 = dt0 / (dt * dt1);

   vector<double> Mq1(b.size(), 0.0);
   vector<double> Mq2(b.size(), 0.0);

   for (int j = 0; j < b.size(); j++)
   {
      Mq1[j] += di_M[j] * prim_field[time_index - 1][j];
      Mq2[j] += di_M[j] * prim_field[time_index - 2][j];
      for (int k = ia[j]; k < ia[j + 1]; k++)
      {
         Mq1[j] += ggl_M[k] * prim_field[time_index - 1][ja[k]];
         Mq1[ja[k]] += ggl_M[k] * prim_field[time_index - 1][j];

         Mq2[j] += ggl_M[k] * prim_field[time_index - 2][ja[k]];
         Mq2[ja[k]] += ggl_M[k] * prim_field[time_index - 2][j];
      }
   }

   for (int i = 0; i < b.size(); i++)
   {
      b[i] += (temp1 * Mq1[i]);
      b[i] -= (temp2 * Mq2[i]);
   }
}

void build_b_anom(direct_task_data& dtd_struct)
{
   int n = dtd_struct.r_coords.size() * dtd_struct.z_coords.size();
   b.resize(n);
   for (int i = 0; i < n; i++) b[i] = 0.0;

   double dt = 0, dt0 = 0, dt1 = 0, temp1 = 0, temp2 = 0, temp0 = 0;
   dt1 = dtd_struct.vec_t[time_index - 1] - dtd_struct.vec_t[time_index - 2];
   dt0 = dtd_struct.vec_t[time_index] - dtd_struct.vec_t[time_index - 1];
   dt = dtd_struct.vec_t[time_index] - dtd_struct.vec_t[time_index - 2];

   temp2 = -dt0 / (dt * dt1);
   temp1 = dt / (dt1 * dt0);

   vector<double> Mq1(n, 0);
   vector<double> Mq2(n, 0);

   for (int j = 0; j < n; j++)
   {
      Mq1[j] += di_M[j] * solutions_vec_t_anom[time_index - 1][j];
      Mq2[j] += di_M[j] * solutions_vec_t_anom[time_index - 2][j];
      for (int k = ia[j]; k < ia[j + 1]; k++)
      {
         Mq1[j] += ggl_M[k] * solutions_vec_t_anom[time_index - 1][ja[k]];
         Mq1[ja[k]] += ggl_M[k] * solutions_vec_t_anom[time_index - 1][j];

         Mq2[j] += ggl_M[k] * solutions_vec_t_anom[time_index - 2][ja[k]];
         Mq2[ja[k]] += ggl_M[k] * solutions_vec_t_anom[time_index - 2][j];
      }
   }

   for (int i = 0; i < n; i++)
   {
      b[i] += (temp1 * Mq1[i]);
      b[i] += (temp2 * Mq2[i]);
   }


   temp2 = dt0 / (dt * dt1);
   temp1 = -dt / (dt1 * dt0);
   temp0 = (dt + dt0) / (dt * dt0);

   vector<double> Mq0_prim(n, 0);
   vector<double> Mq1_prim(n, 0);
   vector<double> Mq2_prim(n, 0);

   for (int j = 0; j < n; j++)
   {
      Mq0_prim[j] += di_M_anom[j] * solutions_vec_t_prim[time_index][j];
      Mq1_prim[j] += di_M_anom[j] * solutions_vec_t_prim[time_index - 1][j];
      Mq2_prim[j] += di_M_anom[j] * solutions_vec_t_prim[time_index - 2][j];
      for (int k = ia[j]; k < ia[j + 1]; k++)
      {
         Mq0_prim[j] += ggl_M_anom[k] * solutions_vec_t_prim[time_index][ja[k]];
         Mq0_prim[ja[k]] += ggl_M_anom[k] * solutions_vec_t_prim[time_index][j];

         Mq1_prim[j] += ggl_M_anom[k] * solutions_vec_t_prim[time_index - 1][ja[k]];
         Mq1_prim[ja[k]] += ggl_M_anom[k] * solutions_vec_t_prim[time_index - 1][j];

         Mq2_prim[j] += ggl_M_anom[k] * solutions_vec_t_prim[time_index - 2][ja[k]];
         Mq2_prim[ja[k]] += ggl_M_anom[k] * solutions_vec_t_prim[time_index - 2][j];
      }
   }

   for (int i = 0; i < n; i++)
   {
      b[i] += (temp0 * Mq0_prim[i]);
      b[i] += (temp1 * Mq1_prim[i]);
      b[i] += (temp2 * Mq2_prim[i]);
   }

}