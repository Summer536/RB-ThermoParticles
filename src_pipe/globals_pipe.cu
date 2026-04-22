#include "pipe.h"

__constant__ double d_pipe_rad1, d_pipe_rad2;
__constant__ double d_pipe_xcenter, d_pipe_ycenter;
__constant__ double d_pipe_omega1,  d_pipe_omega2;
__constant__ int d_npipe;

//PIPES
int npipe = 0;
double pipe_radius1 = PIPE_RAD1, pipe_radius2 = PIPE_RAD2;
double pipe_xcenter = LX / 2.0 pipe_ycenter = LY / 2.0;
double pipe_omega1 = 0.0, pipe_omega2 = PIPE_u0 / PIPE_RAD2;

// PIPE device pointers
int *d_pipe_bnode = nullptr;
int *d_pipe_bnode_owner = nullptr;
PipeLink *d_pipe_links = nullptr; //////////////////
int num_pipe_links = 0;
int *d_pipe_link_count = nullptr;
int *d_pipe_link_offset = nullptr;
void *d_pipe_link_scan_tmp = nullptr;
size_t d_pipe_link_scan_bytes = 0;
int pipe_link_capacity = 0;

double *d_pipe_force_accum_x = nullptr;
double *d_pipe_force_accum_y = nullptr;
double *d_pipe_force_accum_z = nullptr;

// Host pointers definition(particle)
double *h_pipe_force_x = nullptr, *h_pipe_force_y = nullptr, *h_pipe_force_z = nullptr;
double *h_pipe_torque_x = nullptr, *h_pipe_torque_y = nullptr, *h_pipe_torque_z = nullptr;
double *h_pipe_temp = nullptr, *h_pipe_heat = nullptr;

int *h_pipe_bnode = nullptr;
int *h_pipe_bnode_owner = nullptr;

double *h_pipe_force_accum_x = nullptr;
double *h_pipe_force_accum_y = nullptr;
double *h_pipe_force_accum_z = nullptr;
double *h_pipe_torque_accum_x = nullptr;
double *h_pipe_torque_accum_y = nullptr;
double *h_pipe_torque_accum_z = nullptr;

void init_pipe_paras() {

    if (!ACTIVATE_PIPE) {
        npipe = 0;
        return;
    } else {
        npipe = 2;
    }

    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_npipe, &npipe, sizeof(int)));

    // pipe center
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_pipe_xcenter, &pipe_xcenter, sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_pipe_ycenter, &pipe_ycenter, sizeof(double)));

    // pipe angular velocity
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_pipe_omega1, &pipe_omega1, sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_pipe_omega2, &pipe_omega2, sizeof(double)));

}
