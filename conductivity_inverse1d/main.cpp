#include "direct_task.h"
#include "inverse_task.h"

int main()
{
   mesh carcass;
   // get the data we will search for, including the number of layers, their conductivities and heights
   direct_task_data inv_task_experiment_data, dtd_prim;
   get_direct_task_data(inv_task_experiment_data, carcass);
   vector<double> synthetic_eds; // synthetic signal
   // solve the direct task for the received data to get the synthetic signal (eds)
   calculate_prim(inv_task_experiment_data, dtd_prim);
   solve_direct(inv_task_experiment_data, dtd_prim);
   calculate_eds(inv_task_experiment_data, dtd_prim, synthetic_eds);

   inverse_task_data inv_task_input;
   // get initial approximation, regularization vector, acceptor coordinates
   inverse_task_init(inv_task_input, inv_task_experiment_data);
   // for now there's only one acceptor, but usually there are several ones placed as much efficiently as possible
   // to receive enough data to determine the synthetic signal inversion

   inverse_task_solve(inv_task_input, synthetic_eds, carcass, inv_task_experiment_data, dtd_prim);

   return 0;
}