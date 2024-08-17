#ifndef DT_MATRICES_H_SENTRY
#define DT_MATRICES_H_SENTRY

#include "direct_task.h"

void build_portrait();
void build_G();
void build_M();
void build_Mrr();

void build_A_t0();
void build_b_t0(direct_task_data& dtd_struct);

void build_A_t1(direct_task_data& dtd_struct);
void build_b_t1(direct_task_data& dtd_struct);
void build_b_t1_anom(direct_task_data& dtd_struct);

void build_A(direct_task_data& dtd_struct);
void build_b(direct_task_data& dtd_struct);
void build_b_anom(direct_task_data& dtd_struct);

#endif

