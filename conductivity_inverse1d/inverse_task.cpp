#include <iostream>
#include <fstream>

#include "inverse_task.h"
#include "direct_task.h"
#include "solver.h"
#include "dt_functions.h"

extern const double sigma;

void inverse_task_init(inverse_task_data& inv_struct, direct_task_data& experiment_info)
{
   // we will use the same number of layers as there were in the experiment_info
   // (we do not know this information beforehand though in a real experiment though)
   std::ifstream inv_in;
   inv_in.open("inv_input/inv_input.txt");
   if (!inv_in.is_open())
   {
      std::cerr << "could not open file inv_input/inv_input.txt" << std::endl;
      throw std::runtime_error("could not open file inv_input/inv_input.txt");
   }
   double init_sigma = 0.0, init_height = 0.0;
   inv_in >> init_sigma >> init_height;
   inv_in.close();

   inv_struct.init_appr.sigmas.resize(experiment_info.layers_sigma.size() - 1);
   for (int i = 0; i < inv_struct.init_appr.sigmas.size(); i++)
      inv_struct.init_appr.sigmas[i] = init_sigma;
   inv_struct.init_appr.heights.resize(experiment_info.layers_height.size() - 1);
   for (int i = 0; i < inv_struct.init_appr.heights.size(); i++)
      inv_struct.init_appr.heights[i] = init_height;

   inv_struct.reg_vec.sigmas.resize(experiment_info.layers_sigma.size() - 1);
   for (int i = 0; i < inv_struct.reg_vec.sigmas.size(); i++)
      inv_struct.reg_vec.sigmas[i] = init_sigma;
   inv_struct.reg_vec.heights.resize(experiment_info.layers_height.size() - 1);
   for (int i = 0; i < inv_struct.reg_vec.heights.size(); i++)
      inv_struct.reg_vec.heights[i] = init_height;

   inv_struct.acc_r = experiment_info.src_r, inv_struct.acc_z = experiment_info.src_z;
}

double calc_functional(vector<double>& eds_u0, vector<double>& synthetic_eds, vector<double>& weights)
{
   double res = 0.0;
   for (int i = 0; i < synthetic_eds.size(); i++)
   {
      res += weights[i] * weights[i] * (eds_u0[i] - synthetic_eds[i]) * (eds_u0[i] - synthetic_eds[i]);
   }
   return res;
}

void inverse_task_solve(inverse_task_data& inv_struct, vector<double>& synthetic_eds, mesh& carcass,
   direct_task_data& dtd_synth, direct_task_data& dtd_prim) // large and awkward function, to be splitted into smaller ones
{
   vector<double> u0;
   for (int i = 0; i < inv_struct.init_appr.sigmas.size(); i++)
      u0.push_back(inv_struct.init_appr.sigmas[i]);
   for (int i = 0; i < inv_struct.init_appr.heights.size(); i++)
      u0.push_back(inv_struct.init_appr.heights[i]);
   vector<vector<double>> A(u0.size(), vector<double>(u0.size(), 0.0));
   vector<double> f(u0.size(), 0.0);
   vector<double> weights(synthetic_eds.size(), 0.0);
   for (int i = 0; i < weights.size(); i++)
      weights[i] = 1.0 / synthetic_eds[i];

   direct_task_data u0_calc_data;
   vector<vector<double>> eds(u0.size());
   vector<double> alpha_vec(u0.size(), 1e-9), beta(u0.size(), 2e-2); // 7e-2
   vector<double> regv, deltau, u_next, u_new, eds_u0;

   // calculate functional
   //double J_old = 0.0, J_new = 0.0;
   // iterative process
   int it_max = 100;
   for (int it = 0; it < it_max; it++)
   {
      if (it == 0)
      {
         u0.clear();
         for (int i = 0; i < inv_struct.init_appr.sigmas.size(); i++)
            u0.push_back(inv_struct.init_appr.sigmas[i]);
         for (int i = 0; i < inv_struct.init_appr.heights.size(); i++)
            u0.push_back(inv_struct.init_appr.heights[i]);
      }
      dtd_init_from_paramvec(u0_calc_data, u0, carcass, dtd_synth);
      solve_direct(u0_calc_data, dtd_prim);
      calculate_eds(u0_calc_data, dtd_prim, eds_u0);
      //J_old = J_new;
      //J_new = calc_functional(eds_u0, synthetic_eds, weights);
      //std::cout << it << " " << J_new << " ";
      //for (int i = 0; i < u0.size(); i++)
      //   std::cout << u0[i] << " ";
      //std::cout << std::endl;
      for (int i = 0; i < u0.size(); i++)
      {
         vector<double> u_new = u0;
         u_new[i] += 0.05 * u0[i];
         direct_task_data deriv_calc_data;
         dtd_init_from_paramvec(deriv_calc_data, u_new, carcass, dtd_synth);
         solve_direct(deriv_calc_data, dtd_prim);
         calculate_eds(deriv_calc_data, dtd_prim, eds[i]);
      }

      vector<double> u_new = u0;
      for (int i = 0; i < u_new.size(); i++)
         u_new[i] += 0.05 * u0[i];

      for (int i = 0; i < A.size(); i++)
         for (int j = 0; j < A[i].size(); j++)
            A[i][j] = 0.0;
      for (int i = 0; i < f.size(); i++) f[i] = 0.0;
      for (int i = 0; i < A.size(); i++)
      {
         for (int j = 0; j < A[i].size(); j++)
         {
            for (int k = 10; k < synthetic_eds.size() - 10; k++)
            {
               double deriv_i = ((eds[i][k] - synthetic_eds[k]) - (eds_u0[k] - synthetic_eds[k])) / (u_new[i] - u0[i]);
               double deriv_j = ((eds[j][k] - synthetic_eds[k]) - (eds_u0[k] - synthetic_eds[k])) / (u_new[j] - u0[j]);
               A[i][j] += weights[k] * weights[k] * deriv_i * deriv_j;
               f[i] -= weights[k] * weights[k] * (eds_u0[k] - synthetic_eds[k]) * deriv_i;
            }
         }
      }

      // regularization
      if (it == 0)
      {
         regv.clear();
         for (int i = 0; i < inv_struct.reg_vec.sigmas.size(); i++)
            regv.push_back(inv_struct.reg_vec.sigmas[i]);
         for (int i = 0; i < inv_struct.reg_vec.heights.size(); i++)
            regv.push_back(inv_struct.reg_vec.heights[i]);
      }

      // alpha should be computed as min{j}(A[i][j]) * 1e-8 
      for (int i = 0; i < u0.size(); i++)
      {
         A[i][i] += alpha_vec[i];
         f[i] -= alpha_vec[i] * abs(u0[i] - regv[i]);
      }

      deltau.resize(u0.size());
      for (int i = 0; i < deltau.size(); i++) deltau[i] = 0.0;
      gauss_v2(A, deltau, f, f.size(), 1e-15);

      bool beta_reset_flag = true;
      for (int i = 0; i < beta.size(); i++) beta[i] = 1e-1;
      while (beta_reset_flag)
      {
         u_next = u0;
         for (int i = 0; i < u0.size(); i++)
            u_next[i] += beta[i] * deltau[i];

         beta_reset_flag = false;
         for (int i = 0; i < u_next.size(); i++)
         {
            if (u_next[i] <= 0.0)
            {
               for (int j = 0; j < beta.size(); j++)
                  beta[j] /= 2.0;
               beta_reset_flag = true;
            }
         }
         if (beta_reset_flag)
            std::cout << "beta_reset" << std::endl;
      }

      double J = 0.0;
      direct_task_data tmp_calc_data;
      dtd_init_from_paramvec(tmp_calc_data, u_next, carcass, dtd_synth);
      solve_direct(tmp_calc_data, dtd_prim);
      vector<double> eds_tmp;
      calculate_eds(tmp_calc_data, dtd_prim, eds_tmp);
      J = calc_functional(eds_tmp, synthetic_eds, weights);
      std::cout << it << " " << J << " ";
      for (int i = 0; i < u_next.size(); i++)
         std::cout << u_next[i] << " ";
      std::cout << std::endl;

      vector<double> u_next2, u_change;
      int max_decrease_beta = 5, dcr_cnt = 0;
      for (int i = 0; i < max_decrease_beta; i++)
      {
         auto beta2 = beta;
         for (int i = 0; i < beta.size(); i++)
            beta2[i] /= 2.0;
         u_next2 = u0;
         for (int i = 0; i < u0.size(); i++)
            u_next2[i] += beta2[i] * deltau[i];
         bool incorrect = false;
         for (int j = 0; j < u_next2.size(); j++)
            if (u_next2[j] < 0.0)
               incorrect = true;
         if (incorrect) break;
         direct_task_data next2_calc_data;
         dtd_init_from_paramvec(next2_calc_data, u_next2, carcass, dtd_synth);
         solve_direct(next2_calc_data, dtd_prim);
         vector<double> eds_next2;
         calculate_eds(next2_calc_data, dtd_prim, eds_next2);
         double J_tmp = calc_functional(eds_next2, synthetic_eds, weights);
         if (J_tmp < J)
         {
            J = J_tmp;
            std::cout << it << " " << J << " ";
            for (int i = 0; i < u_next2.size(); i++)
               std::cout << u_next2[i] << " ";
            std::cout << std::endl;
            for (int i = 0; i < beta.size(); i++)
               beta[i] /= 2;
            dcr_cnt++;
            u_change = u_next2;
         }
         else
         {
            break;
         }
      }

      int max_increase_beta = 5, incr_cnt = 0;
      for (int i = 0; i < max_increase_beta; i++)
      {
         auto beta2 = beta;
         for (int j = 0; j < beta.size(); j++)
            beta2[j] *= 2.0;
         u_next2 = u0;
         for (int j = 0; j < u0.size(); j++)
            u_next2[j] += beta2[j] * deltau[j];
         bool incorrect = false;
         for (int j = 0; j < u_next2.size(); j++)
            if (u_next2[j] < 0.0)
               incorrect = true;
         if (incorrect) break;
         direct_task_data next2_calc_data;
         dtd_init_from_paramvec(next2_calc_data, u_next2, carcass, dtd_synth);
         solve_direct(next2_calc_data, dtd_prim);
         vector<double> eds_next2;
         calculate_eds(next2_calc_data, dtd_prim, eds_next2);
         double J_tmp = calc_functional(eds_next2, synthetic_eds, weights);
         if (J_tmp < J)
         {
            J = J_tmp;
            std::cout << it << " " << J << " ";
            for (int i = 0; i < u_next2.size(); i++)
               std::cout << u_next2[i] << " ";
            std::cout << std::endl;
            for (int i = 0; i < beta.size(); i++)
               beta[i] *= 2;
            incr_cnt++;
            u_change = u_next2;
         }
         else
         {
            break;
         }
      }

      std::cout << "decrease steps = " << dcr_cnt << ", increase steps = " << incr_cnt << std::endl;


      regv = u0;
      if (dcr_cnt || incr_cnt)
         u0 = u_change;
      else
         u0 = u_next;
   }

   std::cout << "stopped" << std::endl;
}

void dtd_init_from_paramvec(direct_task_data& dtd, vector<double>& u, mesh& carcass, direct_task_data& dtd_synth)
{
   dtd.r_coords = carcass.r_coords;
   dtd.z_coords = carcass.z_coords;
   dtd.layers_sigma.resize(u.size() / 2 + 1);
   for (int i = 0; i < dtd.layers_sigma.size() - 1; i++)
      dtd.layers_sigma[i] = u[i];
   dtd.layers_height.resize(u.size() / 2 + 1);
   for (int i = 0; i < dtd.layers_height.size() - 1; i++)
      dtd.layers_height[i] = u[u.size() / 2 + i];

   double depth = 0.0;
   for (int i = 0; i < dtd.layers_height.size() - 1; i++)
   {
      depth += dtd.layers_height[i];
      dtd.z_coords.push_back(-depth);
   }
   if (depth > -carcass.z_coords[0])
   {
      std::cout << "layer depth is larger than the lower bound" << std::endl;
      std::cout << "max layer height is " << -carcass.z_coords[0] << std::endl;
      exit(-13);
   }
   dtd.layers_height[dtd.layers_height.size() - 1] = (u.back() - carcass.z_coords[0]); // the last layer height
   dtd.layers_sigma[dtd.layers_sigma.size() - 1] = sigma; // default sigma
   // it is not used though
   //dtd.z_coords.push_back(-depth - dtd.layers_height.back()); // not required
   sort_and_remove_duplicates(dtd.z_coords);
   dtd.src_r = dtd_synth.src_r; dtd.src_z = dtd_synth.src_z;
   dtd.vec_t = dtd_synth.vec_t;
}