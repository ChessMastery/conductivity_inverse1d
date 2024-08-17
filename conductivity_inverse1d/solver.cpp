#include "solver.h"

#include "direct_task.h"
#include <vector>
#include <iostream>
#include <string>

double norm_vec_E2(vector<double>& x)
{
   double norm = 0;
   for (int i = 0; i < x.size(); i++)
      norm += x[i] * x[i];
   norm = sqrt(norm);
   return norm;
}

double norm_vec_E2_without_bc1(vector<double>& x, direct_task_data& dtd_struct)
{
   double norm = 0.0;
   for (int i = 1; i < dtd_struct.z_coords.size() - 1; i++)
   {
      for (int j = 1; j < dtd_struct.z_coords.size() - 1; j++)
      {
         norm += x[i * dtd_struct.r_coords.size() + j] * x[i * dtd_struct.r_coords.size() + j];
      }
   }
   norm = sqrt(norm);
   return norm;
}

void los_sparse(vector<double>& ggl, vector<double>& di, vector<int>& ig, vector<int>& jg, vector<double>& x, vector<double>& b, int n)
{
   int max_iter = sqrt(n);
   double eps = 1e-13;

   //if (ig[0]) // in the case an array might be stored in fortran-like format
   //{
   //   for (int i = 0; i < n + 1; i++) ig[i]--;
   //   for (int i = 0; i < ig[n]; i++) jg[i]--;
   //}

   vector<double> pr(n, 0);

   for (int i = 0; i < n; i++)
   {
      pr[i] = b[i];
   }

   // iteration process
   vector<double> r(n, 0);
   for (int i = 0; i < n; i++)
      r[i] = pr[i];
   vector<double> z(n, 0);
   for (int i = 0; i < n; i++)
      z[i] = r[i];
   vector<double> p(n, 0);
   for (int i = 0; i < n; i++)
   {
      p[i] += di[i] * z[i];
      for (int j = ig[i]; j < ig[i + 1]; j++)
      {
         p[i] += ggl[j] * z[jg[j]];
         p[jg[j]] += ggl[j] * z[i];
      }
   }

   //max_iter = 100;
   bool outflag = false;
   int it = 0;
   for (it = 0; it < max_iter && !outflag; it++)
   {
      // computing alpha

      double p_r = 0;
      for (int j = 0; j < n; j++)
         p_r += p[j] * r[j];
      double p_p = 0;
      for (int j = 0; j < n; j++)
         p_p += p[j] * p[j];

      double alpha = p_r / p_p;

      // computing xk
      for (int j = 0; j < n; j++)
         x[j] += alpha * z[j];

      // computing rk
      vector<double> rk(n, 0);
      for (int j = 0; j < n; j++)
         rk[j] = r[j] - alpha * p[j];

      // computing beta

      vector<double> Ark(n, 0);

      for (int i = 0; i < n; i++)
      {
         Ark[i] += di[i] * rk[i];
         for (int j = ig[i]; j < ig[i + 1]; j++)
         {
            Ark[i] += ggl[j] * rk[jg[j]];
            Ark[jg[j]] += ggl[j] * rk[i];
         }
      }

      double p_Ark = 0;
      for (int j = 0; j < n; j++)
         p_Ark += p[j] * Ark[j];
      double beta = -p_Ark / p_p;

      // computing zk
      for (int j = 0; j < n; j++)
         z[j] = rk[j] + beta * z[j];

      // out test

      if (norm_vec_E2(rk) / norm_vec_E2(pr) < eps)
         outflag = true;
      else
         // continue iteration process
         for (int j = 0; j < n; j++)
         {
            p[j] = Ark[j] + beta * p[j];
            r[j] = rk[j];
         }
   }
}

void los_sparse_sst(vector<double>& gg, vector<double>& di, vector<int>& ig, vector<int>& jg,
   vector<double>& x, vector<double>& b, int n, direct_task_data& dtd_struct)
{
   int max_iter = sqrt(n);
   double eps = 1e-15;

   x.resize(n); for (int i = 0; i < n; i++) x[i] = 0.0;

   // matrix factorization (SSt particular)
   vector<double> sdi(n, 0);
   for (int i = 0; i < n; i++) sdi[i] = di[i];
   vector<double> sgg(ig[n], 0);
   for (int i = 0; i < ig[n]; i++) sgg[i] = gg[i];

   sdi[0] = sqrt(sdi[0]);

   for (int i = 1; i < n; i++)
   {
      double sumdi = 0;

      for (int k = ig[i]; k < ig[i + 1]; k++)
      {
         double sum = 0;
         int k_low = ig[jg[k]];
         int k_up = ig[i];
         for (; k_low < ig[jg[k] + 1];)
         {
            if (jg[k_up] == jg[k_low])
            {
               sum += sgg[k_up] * sgg[k_low];
               k_up++;
               k_low++;
            }
            else if (jg[k_up] > jg[k_low])
               k_low++;
            else k_up++;
         }
         sgg[k] = (sgg[k] - sum) / sdi[jg[k]];
      }

      for (int k_di = ig[i]; k_di < ig[i + 1]; k_di++)
         sumdi += sgg[k_di] * sgg[k_di];
      sdi[i] = sqrt(sdi[i] - sumdi);
   }

   // inverse matrix computation (S^(-1)): not required for now

   // iteration process

   for (int i = 0; i < x.size(); i++) x[i] = 0.0;

   vector<double> r(n, 0);
   for (int i = 0; i < n; i++)
   {
      double sum = 0;
      for (int k = ig[i]; k < ig[i + 1]; k++)
         sum += sgg[k] * r[jg[k]];
      r[i] = (b[i] - sum) / sdi[i];
   }

   vector<double> z(n, 0); // backward substitution
   for (int i = n - 1; i >= 0; i--)
   {
      z[i] += r[i] / sdi[i];
      for (int k = ig[i]; k < ig[i + 1]; k++)
         z[jg[k]] -= sgg[k] * z[i] / sdi[jg[k]];
   }

   vector<double> q(n, 0);
   for (int i = 0; i < n; i++)
   {
      q[i] += di[i] * z[i];
      for (int j = ig[i]; j < ig[i + 1]; j++)
      {
         q[i] += gg[j] * z[jg[j]];
         q[jg[j]] += gg[j] * z[i];
      }
   }

   vector<double> p(n, 0); // forward elimination
   for (int i = 0; i < n; i++)
   {
      double sum = 0;
      for (int k = ig[i]; k < ig[i + 1]; k++)
         sum += sgg[k] * p[jg[k]];
      p[i] = (q[i] - sum) / sdi[i];
   }


   bool outflag = false;
   int it = 0;
   for (it = 0; it < max_iter && !outflag; it++)
   {
      // computing alpha
      double p_r = 0;
      for (int j = 0; j < n; j++)
         p_r += p[j] * r[j];
      double p_p = 0;
      for (int j = 0; j < n; j++)
         p_p += p[j] * p[j];
      double alpha = p_r / p_p;

      // computing xk
      for (int j = 0; j < n; j++)
         x[j] += alpha * z[j];

      // computing rk
      vector<double> rk(n, 0);
      for (int j = 0; j < n; j++)
         rk[j] = r[j] - alpha * p[j];

      // computing beta
      vector<double> Urk(n, 0);
      for (int i = n - 1; i >= 0; i--)
      {
         Urk[i] += rk[i] / sdi[i];
         for (int k = ig[i]; k < ig[i + 1]; k++)
            Urk[jg[k]] -= Urk[i] * sgg[k] / sdi[jg[k]];
      }

      vector<double> AUrk(n, 0);
      for (int i = 0; i < n; i++)
      {
         AUrk[i] += di[i] * Urk[i];
         for (int j = ig[i]; j < ig[i + 1]; j++)
         {
            AUrk[i] += gg[j] * Urk[jg[j]];
            AUrk[jg[j]] += gg[j] * Urk[i];
         }
      }

      vector<double> LAUrk(n, 0);
      for (int i = 0; i < n; i++)
      {
         double sum = 0;
         for (int k = ig[i]; k < ig[i + 1]; k++)
            sum += sgg[k] * LAUrk[jg[k]];
         LAUrk[i] = (AUrk[i] - sum) / sdi[i];
      }


      double p_LAUrk = 0;
      for (int j = 0; j < n; j++)
         p_LAUrk += p[j] * LAUrk[j];
      double beta = -p_LAUrk / p_p;

      // computing zk
      for (int j = 0; j < n; j++)
         z[j] = Urk[j] + beta * z[j];

      // exit process test

      /*double rk_rk = 0, rprev_rprev = 0;
      for (int i = 0; i < n; i++)
         rprev_rprev += r[i] * r[i];
      for (int i = 0; i < n; i++)
         rk_rk += rprev_rprev - alpha * alpha * p_p;

      if (rk_rk < eps)
         outflag = true;*/
      if (norm_vec_E2(rk) / norm_vec_E2_without_bc1(b, dtd_struct) < eps)
         outflag = true;

      else
         // continue iteration process
         for (int j = 0; j < n; j++)
         {
            p[j] = LAUrk[j] + beta * p[j];
            r[j] = rk[j];
         }



      //cout << "Iteration number " << it << endl;
      //cout << "norm(rk, n) / norm(p, n): " << norm(rk, n) / norm(pr, n) << endl;
   }

   //auto end_time = chrono::high_resolution_clock::now();
   //double elapsed_time = chrono::duration<double>(end_time - start_time).count();
}

void gauss_v2(vector<vector<double>>& a, vector<double>& x, vector<double>& y, int n, double eps)
{
   int k, index;
   double max;
   k = 0;
   while (k < n)
   {
      max = abs(a[k][k]);
      index = k;
      for (int i = k + 1; i < n; i++)
      {
         if (abs(a[i][k]) > max)
         {
            max = abs(a[i][k]);
            index = i;
         }
      }
      if (max < eps)
      {
         std::cerr << "Gaussian elimination with row pivoting (columns swapping) error: could not find pivot >= eps from column ";
         std::cerr << index << " of the matrix." << std::endl;
         throw std::runtime_error("Gaussian elimination error: could not find pivot >= eps from column" +
            std::to_string(index) + "of the matrix.");
      }
      for (int j = 0; j < n; j++)
      {
         double temp = a[k][j];
         a[k][j] = a[index][j];
         a[index][j] = temp;
      }
      double temp = y[k];
      y[k] = y[index];
      y[index] = temp;

      for (int i = k; i < n; i++)
      {
         double temp = a[i][k];
         if (abs(temp) < eps) continue;
         for (int j = 0; j < n; j++)
            a[i][j] = a[i][j] / temp;
         y[i] = y[i] / temp;
         if (i == k)  continue;
         for (int j = 0; j < n; j++)
            a[i][j] = a[i][j] - a[k][j];
         y[i] = y[i] - y[k];
      }
      k++;
   }
   for (k = n - 1; k >= 0; k--)
   {
      x[k] = y[k];
      for (int i = 0; i < k; i++)
         y[i] = y[i] - a[i][k] * x[k];
   }
}