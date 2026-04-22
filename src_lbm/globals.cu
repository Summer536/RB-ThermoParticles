#include "particle.h"
#include "pipe.h"

__constant__ int d_cix[NPOP];
__constant__ int d_ciy[NPOP];
__constant__ int d_ciz[NPOP];
__constant__ double d_tp[NPOP];
__constant__ int d_opp[NPOP];
__constant__ double d_tau, d_tauc, d_grav0, d_beta, d_tHot, d_tCold, d_t0;
__constant__ double d_diff, d_visc;
__constant__ double d_particle_radius;
__constant__ double d_particle_mass;
__constant__ double d_particle_inertia;
__constant__ double d_particle_heat_capacity;
__constant__ double d_particle_cp_fluid;
__constant__ double d_particle_rho_ratio;
__constant__ int d_npart;

double rayl = RAYLEIGH;
double prand = PRANDTL;
double beta = BETA;
double tHot = T_HOT;
double tCold = T_COLD;
double t0 = T_REF;

double visc = VISC;
double grav0 = (beta != 0.0 && (tHot - tCold) != 0.0)
    ? rayl * visc * visc / (beta * (tHot - tCold) * (double)LX * (double)LX * (double)LX * prand)
    : 0.0;
double tau = 3.0 * visc + 0.5;
double u0 = sqrt(fabs(grav0 * beta * (tHot - tCold) * (double)LX));
double diff = visc / prand;
double tauc = 3.0 * diff + 0.5;
double tauci = 1.0 / tauc;


// Device pointers definition
double *d_f = nullptr, *d_g = nullptr, *d_rho = nullptr, *d_ux = nullptr, *d_uy = nullptr, *d_uz = nullptr, *d_phi = nullptr;
double *d_force_realx = nullptr, *d_force_realy = nullptr, *d_force_realz = nullptr;
double *d_f_temp = nullptr, *d_f_collide = nullptr, *d_g_temp = nullptr, *d_g_collide = nullptr;
double *d_ted = nullptr;
double *d_Ked = nullptr;

// Host pointers definition
double *h_rho = nullptr, *h_ux = nullptr, *h_uy = nullptr, *h_uz = nullptr, *h_phi = nullptr;
double *h_f = nullptr, *h_g = nullptr;
double *h_ted = nullptr, *h_Ked = nullptr;



////////////////////////////////////////////////////////////////////////////////////////////////////

//particles
double particle_radius = 0.0;
double particle_mass = 0.0;
double particle_inertia = 0.0;
double particle_heat_capacity = 0.0;
double particle_cp_fluid = CP_FLUID;
double particle_rho_ratio = RHO_PARTICLE_RATIO;
int restart_step = CONTINUOUS_STEPS;
int npart = 0;
double particle_fraction = PARTICLE_FRACTION;

// Device pointers definition(particle)
double *d_ppos_x = nullptr, *d_ppos_y = nullptr, *d_ppos_z = nullptr;
double *d_pvel_x = nullptr, *d_pvel_y = nullptr, *d_pvel_z = nullptr;
double *d_pomega_x = nullptr, *d_pomega_y = nullptr, *d_pomega_z = nullptr;
double *d_ptheta_x = nullptr, *d_ptheta_y = nullptr, *d_ptheta_z = nullptr;
double *d_pforce_x = nullptr, *d_pforce_y = nullptr, *d_pforce_z = nullptr;
double *d_ptorque_x = nullptr, *d_ptorque_y = nullptr, *d_ptorque_z = nullptr;
double *d_pforce_hist_x = nullptr, *d_pforce_hist_y = nullptr, *d_pforce_hist_z = nullptr;
double *d_ptorque_hist_x = nullptr, *d_ptorque_hist_y = nullptr, *d_ptorque_hist_z = nullptr;
double *d_ptemp = nullptr, *d_pheat = nullptr;
double *d_pheat_prev = nullptr;
int *d_ibnode = nullptr, *d_ibnode_prev = nullptr;
int *d_ibnode_owner = nullptr, *d_ibnode_owner_prev = nullptr;
ParticleLink *d_particle_links = nullptr;
int num_particle_links = 0;
int *d_link_count = nullptr;
int *d_link_offset = nullptr;
void *d_link_scan_tmp = nullptr;
size_t d_link_scan_bytes = 0;
int particle_link_capacity = 0;
int new_fluid_capacity = 0;
int *d_new_fluid_nodes = nullptr, *d_new_fluid_pids = nullptr;
int num_new_fluid_nodes = 0;
double *d_particle_force_accum_x = nullptr;
double *d_particle_force_accum_y = nullptr;
double *d_particle_force_accum_z = nullptr;
double *d_particle_torque_accum_x = nullptr;
double *d_particle_torque_accum_y = nullptr;
double *d_particle_torque_accum_z = nullptr;
double *d_particle_force_prev_x = nullptr;
double *d_particle_force_prev_y = nullptr;
double *d_particle_force_prev_z = nullptr;
double *d_particle_force_prev2_x = nullptr;
double *d_particle_force_prev2_y = nullptr;
double *d_particle_force_prev2_z = nullptr;
double *d_particle_torque_prev_x = nullptr;
double *d_particle_torque_prev_y = nullptr;
double *d_particle_torque_prev_z = nullptr;
double *d_particle_torque_prev2_x = nullptr;
double *d_particle_torque_prev2_y = nullptr;
double *d_particle_torque_prev2_z = nullptr;

// Host pointers definition(particle)
double *h_ppos_x = nullptr, *h_ppos_y = nullptr, *h_ppos_z = nullptr;
double *h_pvel_x = nullptr, *h_pvel_y = nullptr, *h_pvel_z = nullptr;
double *h_pomega_x = nullptr, *h_pomega_y = nullptr, *h_pomega_z = nullptr;
double *h_ptheta_x = nullptr, *h_ptheta_y = nullptr, *h_ptheta_z = nullptr;
double *h_pforce_x = nullptr, *h_pforce_y = nullptr, *h_pforce_z = nullptr;
double *h_ptorque_x = nullptr, *h_ptorque_y = nullptr, *h_ptorque_z = nullptr;
double *h_pforce_hist_x = nullptr, *h_pforce_hist_y = nullptr, *h_pforce_hist_z = nullptr;
double *h_ptorque_hist_x = nullptr, *h_ptorque_hist_y = nullptr, *h_ptorque_hist_z = nullptr;
double *h_ptemp = nullptr, *h_pheat = nullptr;
double *h_pheat_prev = nullptr;
int *h_ibnode = nullptr, *h_ibnode_prev = nullptr;
int *h_ibnode_owner = nullptr, *h_ibnode_owner_prev = nullptr;
int *h_new_fluid_nodes = nullptr, *h_new_fluid_pids = nullptr;
double *h_particle_force_accum_x = nullptr;
double *h_particle_force_accum_y = nullptr;
double *h_particle_force_accum_z = nullptr;
double *h_particle_torque_accum_x = nullptr;
double *h_particle_torque_accum_y = nullptr;
double *h_particle_torque_accum_z = nullptr;

void init_particle_constants() {
    constexpr double pi = 3.14159265358979323846;
    const double radius = 0.5 * PARTICLE_DIAM_FRAC * static_cast<double>(LX);
    const double volume = 4.0 * pi * radius * radius * radius / 3.0;
    const double fluid_density = RHO0;
    const double mass = fluid_density * RHO_PARTICLE_RATIO * volume;
    const double inertia = 0.4 * mass * radius * radius;
    const double heat_capacity = mass * CP_PARTICLE_RATIO * CP_FLUID;

    particle_radius = radius;
    particle_mass = mass;
    particle_inertia = inertia;
    particle_heat_capacity = heat_capacity;
    particle_cp_fluid = CP_FLUID;
    particle_rho_ratio = RHO_PARTICLE_RATIO;

    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_particle_radius, &particle_radius, sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_particle_mass, &particle_mass, sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_particle_inertia, &particle_inertia, sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_particle_heat_capacity, &particle_heat_capacity, sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_particle_cp_fluid, &particle_cp_fluid, sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_particle_rho_ratio, &particle_rho_ratio, sizeof(double)));
}

void update_particle_count() {
    if (!ACTIVATE_PARTICLES) {
        npart = 0;
        return;
    }

    constexpr double pi = 3.14159265358979323846;
    const double radius = 0.5 * PARTICLE_DIAM_FRAC * static_cast<double>(LX);
    const double volume = 4.0 * pi * radius * radius * radius / 3.0;
    if (particle_fraction <= 0.0 || volume <= 0.0) {
        npart = 0;
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_npart, &npart, sizeof(int)));
        return;
    }

    const double raw = LXYZ * particle_fraction / volume;
    const double clamped = (raw < 0.0) ? 0.0 : raw;
    npart = static_cast<int>(floor(clamped));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_npart, &npart, sizeof(int)));
}
