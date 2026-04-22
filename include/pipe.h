#ifndef PIPE_H
#define PIPE_H

#include "lbm.h"

extern int npipe;
extern __constant__ int d_npipe;
// void update_pipe_count();
#ifdef NPIPE
#undef NPIPE
#endif
#ifdef __CUDA_ARCH__
#define NPIPE d_npipe
#else
#define NPIPE npipe
#endif

struct PipeLink {
    int cell_i; //fuild point position x
    int cell_j;
    int cell_k;
    int dir;  //direction ip
    int pid;  //particle ID of link
    double q; //intersection of link and particle boundary (0-1)
    double rx; //q * cix[ip]
    double ry;
    double rz;
};

// Pipe constants (device)
extern __constant__ double d_pipe_rad1, d_pipe_rad2;
extern __constant__ double d_pipe_xcenter, d_pipe_ycenter;
extern __constant__ double d_pipe_omega1,  d_pipe_omega2;

// Pipe globals (host)
// The cylindrical pipe rotates about the z-axis, 
// thus only x-center and y-center is required
extern double pipe_xcenter, pipe_ycenter;
// and the rotating velocity of the pipes are defined in "parameters.h",
// only angular velocity is necessary
extern double pipe_omega1, pipe_omega2;
// Pipe globals (device)


// PIPE device pointer for boundary and domain identifier
extern int *d_pipe_bnode; 
extern int *d_pipe_bnode_owner;
extern PipeLink *d_pipe_links;
extern int num_pipe_links;
extern int *d_pipe_link_count, *d_pipe_link_offset;
extern void *d_pipe_link_scan_tmp;
extern size_t d_pipe_link_scan_bytes;
extern int pipe_link_capacity;

extern double *d_pipe_force_accum_x;
extern double *d_pipe_force_accum_y;
extern double *d_pipe_force_accum_z;

// PIPE host pointers
extern double *h_pipe_force_x, *h_pipe_force_y, *h_pipe_force_z;
extern double *h_pipe_torque_x, *h_pipe_torque_y, *h_pipe_torque_z;
extern double *h_pipe_temp, *h_pipe_heat;
extern int *h_pipe_bnode;
extern int *h_pipe_bnode_owner;


// Particle device pointers
// extern double *d_ppos_x, *d_ppos_y, *d_ppos_z;
// extern double *d_pvel_x, *d_pvel_y, *d_pvel_z;
// extern double *d_pomega_x, *d_pomega_y, *d_pomega_z;
// extern double *d_ptheta_x, *d_ptheta_y, *d_ptheta_z;
extern double *d_pipe_force_x, *d_pipe_force_y, *d_pipe_force_z;
extern double *d_pipe_torque_x, *d_pipe_torque_y, *d_pipe_torque_z;
extern double *d_pipe_temp, *d_pipe_heat;


// Particle host pointers
// extern double *h_ppos_x, *h_ppos_y, *h_ppos_z;
// extern double *h_pvel_x, *h_pvel_y, *h_pvel_z;
// extern double *h_pomega_x, *h_pomega_y, *h_pomega_z;
// extern double *h_ptheta_x, *h_ptheta_y, *h_ptheta_z;
extern double *h_pipe_force_x, *h_pipe_force_y, *h_pipe_force_z;
extern double *h_ptorque_x, *h_pipe_torque_y, *h_pipe_torque_z;
extern double *h_pipe_temp, *h_pipe_heat;
extern int *h_pipe_bnode;
extern int *h_pipe_bnode_owner;
// extern int *h_new_fluid_nodes, *h_new_fluid_pids;
// extern double *h_particle_force_accum_x;
// extern double *h_particle_force_accum_y;
// extern double *h_particle_force_accum_z;
// extern double *h_particle_torque_accum_x;
// extern double *h_particle_torque_accum_y;
// extern double *h_particle_torque_accum_z;

// Pipe API
void init_pipe_paras();
void init_pipes();

void build_pipe_links();
void pipe_bounce_back();
void pipe_bounce_back_thermal();

template <typename T>
inline T* malloc_host_array(size_t n) {
    T* ptr = static_cast<T*>(malloc(n * sizeof(T)));
    if (!ptr) {
        fprintf(stderr, "Failed to allocate particle host buffer\n");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

#endif
