#ifndef PARTICLE_H
#define PARTICLE_H

#include "lbm.h"

extern int npart;
extern double particle_fraction;
extern __constant__ int d_npart;
void update_particle_count();
#ifdef NPART
#undef NPART
#endif
#ifdef __CUDA_ARCH__
#define NPART d_npart
#else
#define NPART npart
#endif

struct ParticleLink {
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

// Particle constants (device)
extern __constant__ double d_particle_radius;
extern __constant__ double d_particle_mass;
extern __constant__ double d_particle_inertia;
extern __constant__ double d_particle_heat_capacity;
extern __constant__ double d_particle_cp_fluid;
extern __constant__ double d_particle_rho_ratio;

// Particle globals (host)
extern double particle_radius;
extern double particle_mass;
extern double particle_inertia;
extern double particle_heat_capacity;
extern double particle_cp_fluid;
extern double particle_rho_ratio;

// Particle device pointers
extern double *d_ppos_x, *d_ppos_y, *d_ppos_z;
extern double *d_pvel_x, *d_pvel_y, *d_pvel_z;
extern double *d_pomega_x, *d_pomega_y, *d_pomega_z;
extern double *d_ptheta_x, *d_ptheta_y, *d_ptheta_z;
extern double *d_pforce_x, *d_pforce_y, *d_pforce_z;
extern double *d_ptorque_x, *d_ptorque_y, *d_ptorque_z;
extern double *d_pforce_hist_x, *d_pforce_hist_y, *d_pforce_hist_z;
extern double *d_ptorque_hist_x, *d_ptorque_hist_y, *d_ptorque_hist_z;
extern double *d_ptemp, *d_pheat;
extern double *d_pheat_prev;
extern int *d_ibnode, *d_ibnode_prev;
extern int *d_ibnode_owner, *d_ibnode_owner_prev;
extern ParticleLink *d_particle_links;
extern int num_particle_links;
extern int *d_link_count, *d_link_offset;
extern void *d_link_scan_tmp;
extern size_t d_link_scan_bytes;
extern int particle_link_capacity;
extern int new_fluid_capacity;
extern int *d_new_fluid_nodes, *d_new_fluid_pids;
extern int num_new_fluid_nodes;
extern double *d_particle_force_accum_x;
extern double *d_particle_force_accum_y;
extern double *d_particle_force_accum_z;
extern double *d_particle_torque_accum_x;
extern double *d_particle_torque_accum_y;
extern double *d_particle_torque_accum_z;
extern double *d_particle_force_prev_x;
extern double *d_particle_force_prev_y;
extern double *d_particle_force_prev_z;
extern double *d_particle_force_prev2_x;
extern double *d_particle_force_prev2_y;
extern double *d_particle_force_prev2_z;
extern double *d_particle_torque_prev_x;
extern double *d_particle_torque_prev_y;
extern double *d_particle_torque_prev_z;
extern double *d_particle_torque_prev2_x;
extern double *d_particle_torque_prev2_y;
extern double *d_particle_torque_prev2_z;

// Particle host pointers
extern double *h_ppos_x, *h_ppos_y, *h_ppos_z;
extern double *h_pvel_x, *h_pvel_y, *h_pvel_z;
extern double *h_pomega_x, *h_pomega_y, *h_pomega_z;
extern double *h_ptheta_x, *h_ptheta_y, *h_ptheta_z;
extern double *h_pforce_x, *h_pforce_y, *h_pforce_z;
extern double *h_ptorque_x, *h_ptorque_y, *h_ptorque_z;
extern double *h_pforce_hist_x, *h_pforce_hist_y, *h_pforce_hist_z;
extern double *h_ptorque_hist_x, *h_ptorque_hist_y, *h_ptorque_hist_z;
extern double *h_ptemp, *h_pheat;
extern double *h_pheat_prev;
extern int *h_ibnode, *h_ibnode_prev;
extern int *h_ibnode_owner, *h_ibnode_owner_prev;
extern int *h_new_fluid_nodes, *h_new_fluid_pids;
extern double *h_particle_force_accum_x;
extern double *h_particle_force_accum_y;
extern double *h_particle_force_accum_z;
extern double *h_particle_torque_accum_x;
extern double *h_particle_torque_accum_y;
extern double *h_particle_torque_accum_z;

// Particle API
void init_particle_constants();
void init_particles();

void build_links();
void refill_nodes();
void bounce_back_particles();
void bounce_back_thermal_particles();
void prepare_particle_forces_step();
void compute_particle_forces();
void compute_particle_heat();
void apply_repulsive_forces();
void update_particles();
void output_particles(int istep);

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
