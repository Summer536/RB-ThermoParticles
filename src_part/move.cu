#include "particle.h"

namespace {

__global__ void update_particles_kernel(double *ppos_x, double *ppos_y, double *ppos_z,
                                        double *pvel_x, double *pvel_y, double *pvel_z,
                                        double *pomega_x, double *pomega_y, double *pomega_z,
                                        double *ptheta_x, double *ptheta_y, double *ptheta_z,
                                        const double *force_x, const double *force_y, const double *force_z,
                                        const double *torque_x, const double *torque_y, const double *torque_z,
                                        const double *force_prev_x, const double *force_prev_y,
                                        const double *force_prev_z,
                                        const double *force_prev2_x, const double *force_prev2_y,
                                        const double *force_prev2_z,
                                        const double *torque_prev_x, const double *torque_prev_y,
                                        const double *torque_prev_z,
                                        const double *torque_prev2_x, const double *torque_prev2_y,
                                        const double *torque_prev2_z,
                                        double *ptemp, const double *pheat,
                                        const double *pheat_prev,
                                        double radius, double mass, double inertia,
                                        double heat_capacity, double rho_ratio) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= NPART) return;
    
    const double dt = 1.0;
    const double forceX = force_x[pid];
    const double forceY = force_y[pid];
    const double forceZ = force_z[pid];
    const double torqueX = torque_x[pid];
    const double torqueY = torque_y[pid];
    const double torqueZ = torque_z[pid];

    const double fx_prev = force_prev_x[pid];
    const double fy_prev = force_prev_y[pid];
    const double fz_prev = force_prev_z[pid];
    const double fx_prev2 = force_prev2_x[pid];
    const double fy_prev2 = force_prev2_y[pid];
    const double fz_prev2 = force_prev2_z[pid];
    const double torque_prev_x_val = torque_prev_x[pid];
    const double torque_prev_y_val = torque_prev_y[pid];
    const double torque_prev_z_val = torque_prev_z[pid];
    const double torque_prev2_x_val = torque_prev2_x[pid];
    const double torque_prev2_y_val = torque_prev2_y[pid];
    const double torque_prev2_z_val = torque_prev2_z[pid];

    const double buoyancy = (1.0 - 1.0 / rho_ratio) * d_grav0;

    const double vx_old = pvel_x[pid];
    const double vy_old = pvel_y[pid];
    const double vz_old = pvel_z[pid];
    const double omega_x_old = pomega_x[pid];
    const double omega_y_old = pomega_y[pid];
    const double omega_z_old = pomega_z[pid];
    
    const double combo_fx = 0.5 * forceX + fx_prev + 0.5 * fx_prev2;
    const double combo_fy = 0.5 * forceY + fy_prev + 0.5 * fy_prev2;
    const double combo_fz = 0.5 * forceZ + fz_prev + 0.5 * fz_prev2;
    const double combo_torque_x = 0.5 * torqueX + torque_prev_x_val + 0.5 * torque_prev2_x_val;
    const double combo_torque_y = 0.5 * torqueY + torque_prev_y_val + 0.5 * torque_prev2_y_val;
    const double combo_torque_z = 0.5 * torqueZ + torque_prev_z_val + 0.5 * torque_prev2_z_val;

    double vx_new = vx_old + 0.5 * dt * combo_fx / mass + dt * buoyancy;   //velocity
    double vy_new = vy_old + 0.5 * dt * combo_fy / mass;
    double vz_new = vz_old + 0.5 * dt * combo_fz / mass;
    double omega_x_new = omega_x_old + 0.5 * dt * combo_torque_x / inertia;
    double omega_y_new = omega_y_old + 0.5 * dt * combo_torque_y / inertia;
    double omega_z_new = omega_z_old + 0.5 * dt * combo_torque_z / inertia;

    double x_old = ppos_x[pid];
    double y_old = ppos_y[pid];
    double z_old = ppos_z[pid];
    double x_new = x_old + 0.5 * dt * (vx_new + vx_old);  
    double y_new = y_old + 0.5 * dt * (vy_new + vy_old);
    double z_new = z_old + 0.5 * dt * (vz_new + vz_old);

    ppos_x[pid] = x_new;  //position
    ppos_y[pid] = y_new;
    ppos_z[pid] = z_new;
    pvel_x[pid] = vx_new;
    pvel_y[pid] = vy_new;
    pvel_z[pid] = vz_new;

    pomega_x[pid] = omega_x_new;
    pomega_y[pid] = omega_y_new;
    pomega_z[pid] = omega_z_new;

    double theta_x = ptheta_x[pid];
    double theta_y = ptheta_y[pid];
    double theta_z = ptheta_z[pid];
    theta_x += 0.5 * dt * (omega_x_new + omega_x_old); //angle
    theta_y += 0.5 * dt * (omega_y_new + omega_y_old);
    theta_z += 0.5 * dt * (omega_z_new + omega_z_old);
    ptheta_x[pid] = theta_x;
    ptheta_y[pid] = theta_y;
    ptheta_z[pid] = theta_z;

    const double heat = pheat[pid];
    const double heat_prev = pheat_prev[pid];
    double T = ptemp[pid];
    T += 0.5 * dt * (heat + heat_prev) / heat_capacity;   //Tempreatrue
    ptemp[pid] = T;
}

} // namespace

void update_particles() {
    if (!ACTIVATE_PARTICLES || NPART == 0) return;
    const int threads = 128;
    const int blocks = (NPART + threads - 1) / threads;

    update_particles_kernel<<<blocks, threads>>>(d_ppos_x, d_ppos_y, d_ppos_z,
                                                 d_pvel_x, d_pvel_y, d_pvel_z,
                                                 d_pomega_x, d_pomega_y, d_pomega_z,
                                                 d_ptheta_x, d_ptheta_y, d_ptheta_z,
                                                 d_particle_force_accum_x,
                                                 d_particle_force_accum_y,
                                                 d_particle_force_accum_z,
                                                 d_particle_torque_accum_x,
                                                 d_particle_torque_accum_y,
                                                 d_particle_torque_accum_z,
                                                 d_particle_force_prev_x,
                                                 d_particle_force_prev_y,
                                                 d_particle_force_prev_z,
                                                 d_particle_force_prev2_x,
                                                 d_particle_force_prev2_y,
                                                 d_particle_force_prev2_z,
                                                 d_particle_torque_prev_x,
                                                 d_particle_torque_prev_y,
                                                 d_particle_torque_prev_z,
                                                 d_particle_torque_prev2_x,
                                                 d_particle_torque_prev2_y,
                                                 d_particle_torque_prev2_z,
                                                 d_ptemp, d_pheat,
                                                 d_pheat_prev,
                                                 particle_radius,
                                                 particle_mass,
                                                 particle_inertia,
                                                 particle_heat_capacity,
                                                 particle_rho_ratio);
    CHECK_CUDA_ERROR(cudaGetLastError());
}
