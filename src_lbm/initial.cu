#include "particle.h"
#include "pipe.h"
#include <vector>
#include <fstream>

namespace {

void load_distributions_from_restart(const char *dirname, int step) {
    char fg_filename[512];
    snprintf(fg_filename, sizeof(fg_filename), "%s/fg%09d.dat", dirname, step);

    FILE *fp = fopen(fg_filename, "rb");
    if (!fp) {
        fprintf(stderr, "Failed to open restart distribution file %s\n", fg_filename);
        exit(EXIT_FAILURE);
    }

    size_t total = static_cast<size_t>(NPOP) * static_cast<size_t>(LXYZ);
    size_t read_f = fread(h_f, sizeof(double), total, fp);
    size_t read_g = fread(h_g, sizeof(double), total, fp);
    fclose(fp);

    if (read_f != total || read_g != total) {
        fprintf(stderr, "Restart distribution file %s is corrupted (read %zu/%zu for f, %zu/%zu for g)\n",
                fg_filename, read_f, total, read_g, total);
        exit(EXIT_FAILURE);
    }

    CHECK_CUDA_ERROR(cudaMemcpy(d_f, h_f, total * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_g, h_g, total * sizeof(double), cudaMemcpyHostToDevice));
}

void load_flow_fields(const char *dirname, int step) {
    char flow_filename[512];
    snprintf(flow_filename, sizeof(flow_filename), "%s/%09d.dat", dirname, step);
    FILE *fp = fopen(flow_filename, "rb");
    if (!fp) {
        fprintf(stderr, "Failed to open flow restart file %s\n", flow_filename);
        exit(EXIT_FAILURE);
    }

    const size_t count = static_cast<size_t>(LXYZ);
    double *host_fields[7] = {h_rho, h_ux, h_uy, h_uz, h_phi, h_ted, h_Ked};
    for (int k = 0; k < 7; ++k) {
        fread(host_fields[k], sizeof(double), count, fp);
    }
    fclose(fp);

    CHECK_CUDA_ERROR(cudaMemcpy(d_rho, h_rho, count * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_ux,  h_ux,  count * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_uy,  h_uy,  count * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_uz,  h_uz,  count * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_phi, h_phi, count * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_ted, h_ted, count * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_Ked, h_Ked, count * sizeof(double), cudaMemcpyHostToDevice));
}

// void load_particles_from_restart(const char *dirname, int step) {
//     if (!(ACTIVATE_PARTICLES && NPART > 0)) return;

//     char pos_filename[1024];
//     snprintf(pos_filename, sizeof(pos_filename), "%s/pos/pos_%09d.txt", dirname, step);

//     std::ifstream fin(pos_filename);
//     if (!fin) {
//         fprintf(stderr, "Failed to open restart particle file %s\n", pos_filename);
//         exit(EXIT_FAILURE);
//     }

//     std::vector<double> force_prev_x(NPART, 0.0), force_prev_y(NPART, 0.0), force_prev_z(NPART, 0.0);
//     std::vector<double> force_prev2_x(NPART, 0.0), force_prev2_y(NPART, 0.0), force_prev2_z(NPART, 0.0);
//     std::vector<double> torque_prev_x(NPART, 0.0), torque_prev_y(NPART, 0.0), torque_prev_z(NPART, 0.0);
//     std::vector<double> torque_prev2_x(NPART, 0.0), torque_prev2_y(NPART, 0.0), torque_prev2_z(NPART, 0.0);

//     for (int lines_read = 0; lines_read < NPART; ++lines_read) {
//         int pid = -1;
//         double px = 0.0, py = 0.0, pz = 0.0;
//         double vx = 0.0, vy = 0.0, vz = 0.0;
//         double fx = 0.0, fy = 0.0, fz = 0.0;
//         double fx_prev = 0.0, fy_prev = 0.0, fz_prev = 0.0;
//         double fx_prev2 = 0.0, fy_prev2 = 0.0, fz_prev2 = 0.0;
//         double theta_x = 0.0, theta_y = 0.0, theta_z = 0.0;
//         double omega_x = 0.0, omega_y = 0.0, omega_z = 0.0;
//         double torque_x = 0.0, torque_y = 0.0, torque_z = 0.0;
//         double torque_prev_x_val = 0.0, torque_prev_y_val = 0.0, torque_prev_z_val = 0.0;
//         double torque_prev2_x_val = 0.0, torque_prev2_y_val = 0.0, torque_prev2_z_val = 0.0;
//         double temp = 0.0, heat = 0.0, hprev = 0.0;

//         if (!(fin >> pid
//               >> px >> py >> pz
//               >> vx >> vy >> vz
//               >> fx >> fy >> fz
//               >> fx_prev >> fy_prev >> fz_prev
//               >> fx_prev2 >> fy_prev2 >> fz_prev2
//               >> theta_x >> theta_y >> theta_z
//               >> omega_x >> omega_y >> omega_z
//               >> torque_x >> torque_y >> torque_z
//               >> torque_prev_x_val >> torque_prev_y_val >> torque_prev_z_val
//               >> torque_prev2_x_val >> torque_prev2_y_val >> torque_prev2_z_val
//               >> temp >> heat >> hprev)) {
//             fprintf(stderr, "Failed to read particle restart data from %s\n", pos_filename);
//             exit(EXIT_FAILURE);
//         }

//         h_ppos_x[pid] = px;
//         h_ppos_y[pid] = py;
//         h_ppos_z[pid] = pz;
//         h_pvel_x[pid] = vx;
//         h_pvel_y[pid] = vy;
//         h_pvel_z[pid] = vz;

//         h_pforce_x[pid] = fx;
//         h_pforce_y[pid] = fy;
//         h_pforce_z[pid] = fz;
//         h_particle_force_accum_x[pid] = fx;
//         h_particle_force_accum_y[pid] = fy;
//         h_particle_force_accum_z[pid] = fz;
//         force_prev_x[pid] = fx_prev;
//         force_prev_y[pid] = fy_prev;
//         force_prev_z[pid] = fz_prev;
//         force_prev2_x[pid] = fx_prev2;
//         force_prev2_y[pid] = fy_prev2;
//         force_prev2_z[pid] = fz_prev2;

//         h_ptheta_x[pid] = theta_x;
//         h_ptheta_y[pid] = theta_y;
//         h_ptheta_z[pid] = theta_z;
//         h_pomega_x[pid] = omega_x;
//         h_pomega_y[pid] = omega_y;
//         h_pomega_z[pid] = omega_z;

//         h_ptorque_x[pid] = torque_x;
//         h_ptorque_y[pid] = torque_y;
//         h_ptorque_z[pid] = torque_z;
//         h_particle_torque_accum_x[pid] = torque_x;
//         h_particle_torque_accum_y[pid] = torque_y;
//         h_particle_torque_accum_z[pid] = torque_z;
//         torque_prev_x[pid] = torque_prev_x_val;
//         torque_prev_y[pid] = torque_prev_y_val;
//         torque_prev_z[pid] = torque_prev_z_val;
//         torque_prev2_x[pid] = torque_prev2_x_val;
//         torque_prev2_y[pid] = torque_prev2_y_val;
//         torque_prev2_z[pid] = torque_prev2_z_val;

//         h_ptemp[pid] = temp;
//         h_pheat[pid] = heat;
//         h_pheat_prev[pid] = hprev;
//     }
//     fin.close();

//     CHECK_CUDA_ERROR(cudaMemcpy(d_ppos_x, h_ppos_x, NPART * sizeof(double), cudaMemcpyHostToDevice));
//     CHECK_CUDA_ERROR(cudaMemcpy(d_ppos_y, h_ppos_y, NPART * sizeof(double), cudaMemcpyHostToDevice));
//     CHECK_CUDA_ERROR(cudaMemcpy(d_ppos_z, h_ppos_z, NPART * sizeof(double), cudaMemcpyHostToDevice));
//     CHECK_CUDA_ERROR(cudaMemcpy(d_pvel_x, h_pvel_x, NPART * sizeof(double), cudaMemcpyHostToDevice));
//     CHECK_CUDA_ERROR(cudaMemcpy(d_pvel_y, h_pvel_y, NPART * sizeof(double), cudaMemcpyHostToDevice));
//     CHECK_CUDA_ERROR(cudaMemcpy(d_pvel_z, h_pvel_z, NPART * sizeof(double), cudaMemcpyHostToDevice));
//     CHECK_CUDA_ERROR(cudaMemcpy(d_pforce_x, h_pforce_x, NPART * sizeof(double), cudaMemcpyHostToDevice));
//     CHECK_CUDA_ERROR(cudaMemcpy(d_pforce_y, h_pforce_y, NPART * sizeof(double), cudaMemcpyHostToDevice));
//     CHECK_CUDA_ERROR(cudaMemcpy(d_pforce_z, h_pforce_z, NPART * sizeof(double), cudaMemcpyHostToDevice));
//     CHECK_CUDA_ERROR(cudaMemcpy(d_particle_force_accum_x, h_particle_force_accum_x, NPART * sizeof(double), cudaMemcpyHostToDevice));
//     CHECK_CUDA_ERROR(cudaMemcpy(d_particle_force_accum_y, h_particle_force_accum_y, NPART * sizeof(double), cudaMemcpyHostToDevice));
//     CHECK_CUDA_ERROR(cudaMemcpy(d_particle_force_accum_z, h_particle_force_accum_z, NPART * sizeof(double), cudaMemcpyHostToDevice));
//     CHECK_CUDA_ERROR(cudaMemcpy(d_pomega_x, h_pomega_x, NPART * sizeof(double), cudaMemcpyHostToDevice));
//     CHECK_CUDA_ERROR(cudaMemcpy(d_pomega_y, h_pomega_y, NPART * sizeof(double), cudaMemcpyHostToDevice));
//     CHECK_CUDA_ERROR(cudaMemcpy(d_pomega_z, h_pomega_z, NPART * sizeof(double), cudaMemcpyHostToDevice));
//     CHECK_CUDA_ERROR(cudaMemcpy(d_ptheta_x, h_ptheta_x, NPART * sizeof(double), cudaMemcpyHostToDevice));
//     CHECK_CUDA_ERROR(cudaMemcpy(d_ptheta_y, h_ptheta_y, NPART * sizeof(double), cudaMemcpyHostToDevice));
//     CHECK_CUDA_ERROR(cudaMemcpy(d_ptheta_z, h_ptheta_z, NPART * sizeof(double), cudaMemcpyHostToDevice));
//     CHECK_CUDA_ERROR(cudaMemcpy(d_ptorque_x, h_ptorque_x, NPART * sizeof(double), cudaMemcpyHostToDevice));
//     CHECK_CUDA_ERROR(cudaMemcpy(d_ptorque_y, h_ptorque_y, NPART * sizeof(double), cudaMemcpyHostToDevice));
//     CHECK_CUDA_ERROR(cudaMemcpy(d_ptorque_z, h_ptorque_z, NPART * sizeof(double), cudaMemcpyHostToDevice));
//     CHECK_CUDA_ERROR(cudaMemcpy(d_particle_torque_accum_x, h_particle_torque_accum_x, NPART * sizeof(double), cudaMemcpyHostToDevice));
//     CHECK_CUDA_ERROR(cudaMemcpy(d_particle_torque_accum_y, h_particle_torque_accum_y, NPART * sizeof(double), cudaMemcpyHostToDevice));
//     CHECK_CUDA_ERROR(cudaMemcpy(d_particle_torque_accum_z, h_particle_torque_accum_z, NPART * sizeof(double), cudaMemcpyHostToDevice));
//     CHECK_CUDA_ERROR(cudaMemcpy(d_ptemp, h_ptemp, NPART * sizeof(double), cudaMemcpyHostToDevice));
//     CHECK_CUDA_ERROR(cudaMemcpy(d_pheat, h_pheat, NPART * sizeof(double), cudaMemcpyHostToDevice));
//     CHECK_CUDA_ERROR(cudaMemcpy(d_pheat_prev, h_pheat_prev, NPART * sizeof(double), cudaMemcpyHostToDevice));

//     CHECK_CUDA_ERROR(cudaMemcpy(d_particle_force_prev_x, force_prev_x.data(), NPART * sizeof(double), cudaMemcpyHostToDevice));
//     CHECK_CUDA_ERROR(cudaMemcpy(d_particle_force_prev_y, force_prev_y.data(), NPART * sizeof(double), cudaMemcpyHostToDevice));
//     CHECK_CUDA_ERROR(cudaMemcpy(d_particle_force_prev_z, force_prev_z.data(), NPART * sizeof(double), cudaMemcpyHostToDevice));
//     CHECK_CUDA_ERROR(cudaMemcpy(d_particle_force_prev2_x, force_prev2_x.data(), NPART * sizeof(double), cudaMemcpyHostToDevice));
//     CHECK_CUDA_ERROR(cudaMemcpy(d_particle_force_prev2_y, force_prev2_y.data(), NPART * sizeof(double), cudaMemcpyHostToDevice));
//     CHECK_CUDA_ERROR(cudaMemcpy(d_particle_force_prev2_z, force_prev2_z.data(), NPART * sizeof(double), cudaMemcpyHostToDevice));
//     CHECK_CUDA_ERROR(cudaMemcpy(d_particle_torque_prev_x, torque_prev_x.data(), NPART * sizeof(double), cudaMemcpyHostToDevice));
//     CHECK_CUDA_ERROR(cudaMemcpy(d_particle_torque_prev_y, torque_prev_y.data(), NPART * sizeof(double), cudaMemcpyHostToDevice));
//     CHECK_CUDA_ERROR(cudaMemcpy(d_particle_torque_prev_z, torque_prev_z.data(), NPART * sizeof(double), cudaMemcpyHostToDevice));
//     CHECK_CUDA_ERROR(cudaMemcpy(d_particle_torque_prev2_x, torque_prev2_x.data(), NPART * sizeof(double), cudaMemcpyHostToDevice));
//     CHECK_CUDA_ERROR(cudaMemcpy(d_particle_torque_prev2_y, torque_prev2_y.data(), NPART * sizeof(double), cudaMemcpyHostToDevice));
//     CHECK_CUDA_ERROR(cudaMemcpy(d_particle_torque_prev2_z, torque_prev2_z.data(), NPART * sizeof(double), cudaMemcpyHostToDevice));

//     build_links();
// }

void load_restart_state(int step, dim3 grid, dim3 block) {
    char dirname[256];
    snprintf(dirname, sizeof(dirname), "Ra%.1ePr%.2f", rayl, prand);

    load_distributions_from_restart(dirname, step);
    // if (ACTIVATE_PARTICLES && NPART > 0) {
    //     load_particles_from_restart(dirname, step);
    // }

    CHECK_CUDA_ERROR(cudaMemcpy(d_f_collide, d_f, NPOP * LXYZ * sizeof(double), cudaMemcpyDeviceToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_g_collide, d_g, NPOP * LXYZ * sizeof(double), cudaMemcpyDeviceToDevice));

    load_flow_fields(dirname, step);
    forcing<<<grid, block>>>(d_force_realx, d_force_realy, d_force_realz, d_phi);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    if (ACTIVATE_PARTICLES && NPART > 0) {
        CHECK_CUDA_ERROR(cudaMemcpy(h_ibnode, d_ibnode, LXYZ * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(h_ibnode_owner, d_ibnode_owner, LXYZ * sizeof(int), cudaMemcpyDeviceToHost));
        std::copy(h_ibnode, h_ibnode + LXYZ, h_ibnode_prev);
        std::copy(h_ibnode_owner, h_ibnode_owner + LXYZ, h_ibnode_owner_prev);
        CHECK_CUDA_ERROR(cudaMemcpy(d_ibnode_prev, h_ibnode_prev, LXYZ * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_ibnode_owner_prev, h_ibnode_owner_prev, LXYZ * sizeof(int), cudaMemcpyHostToDevice));
        num_new_fluid_nodes = 0;
        if (d_new_fluid_nodes) {
            CHECK_CUDA_ERROR(cudaFree(d_new_fluid_nodes));
            d_new_fluid_nodes = nullptr;
        }
        if (d_new_fluid_pids) {
            CHECK_CUDA_ERROR(cudaFree(d_new_fluid_pids));
            d_new_fluid_pids = nullptr;
        }
        if (h_new_fluid_nodes) {
            free(h_new_fluid_nodes);
            h_new_fluid_nodes = nullptr;
        }
        if (h_new_fluid_pids) {
            free(h_new_fluid_pids);
            h_new_fluid_pids = nullptr;
        }
    }
}

} // namespace

void initialize() {
    // if (ACTIVATE_PARTICLES) {
    //     update_particle_count();
    // }
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_f,      NPOP * LXYZ * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_g,      NPOP * LXYZ * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_f_temp, NPOP * LXYZ * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_f_collide, NPOP * LXYZ * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_g_temp, NPOP * LXYZ * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_g_collide, NPOP * LXYZ * sizeof(double)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_rho,         LXYZ * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_ux,          LXYZ * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_uy,          LXYZ * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_uz,          LXYZ * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_phi,         LXYZ * sizeof(double)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_force_realx, LXYZ * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_force_realy, LXYZ * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_force_realz, LXYZ * sizeof(double)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_ted,         LXYZ * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_Ked,         LXYZ * sizeof(double)));

    h_rho = (double*)malloc(LXYZ * sizeof(double));
    h_ux  = (double*)malloc(LXYZ * sizeof(double));
    h_uy  = (double*)malloc(LXYZ * sizeof(double));
    h_uz  = (double*)malloc(LXYZ * sizeof(double));
    h_phi = (double*)malloc(LXYZ * sizeof(double));
    h_f   = (double*)malloc(NPOP * LXYZ * sizeof(double));
    h_g   = (double*)malloc(NPOP * LXYZ * sizeof(double));
    h_ted = (double*)malloc(LXYZ * sizeof(double));
    h_Ked = (double*)malloc(LXYZ * sizeof(double));

    // Initialize constant memory (D3Q27)
    int h_cix[NPOP] = {0,  1,  0, -1,  0,  0,  0,  1, -1, -1,  1,  1,  0, -1,  0,  1,  0, -1,  0,  1, -1, -1,  1,  1, -1, -1,  1};
    int h_ciy[NPOP] = {0,  0,  1,  0, -1,  0,  0,  1,  1, -1, -1,  0,  1,  0, -1,  0,  1,  0, -1,  1,  1, -1, -1,  1,  1, -1, -1};
    int h_ciz[NPOP] = {0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  1,  1,  1,  1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1};
    int h_opp[NPOP] = {0,  3,  4,  1,  2,  6,  5,  9, 10,  7,  8, 17, 18, 15, 16, 13, 14, 11, 12, 25, 26, 23, 24, 21, 22, 19, 20};

    double h_tp[NPOP];
    for (int i = 0; i < NPOP; i++) {
        if (i == 0)          h_tp[i] = 8.0 / 27.0;
        else if (i <= 6)     h_tp[i] = 2.0 / 27.0;
        else if (i <= 18)    h_tp[i] = 1.0 / 54.0;
        else                 h_tp[i] = 1.0 / 216.0;
    }

    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_cix,   h_cix, NPOP * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_ciy,   h_ciy, NPOP * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_ciz,   h_ciz, NPOP * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_tp,    h_tp,  NPOP * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_opp,   h_opp, NPOP * sizeof(int)));

    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_tau,   &tau,   sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_tauc,  &tauc,  sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_grav0, &grav0, sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_beta,  &beta,  sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_tHot,  &tHot,  sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_tCold, &tCold, sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_t0,    &t0,    sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_diff,  &diff,  sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_visc,  &visc,  sizeof(double)));

    bool use_restart = (restart_step > 0);

    if (ACTIVATE_PIPE && NPIPE > 0) {
        init_pipes();
    }
    dim3 grid((LX + BLOCK_X - 1) / BLOCK_X,
              (LY + BLOCK_Y - 1) / BLOCK_Y,
              (LZ + BLOCK_Z - 1) / BLOCK_Z);
    dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);

    if (use_restart) {
        load_restart_state(restart_step, grid, block);
    } else {
        init_f<<<grid, block>>>(d_f, d_rho, d_ux, d_uy, d_uz);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        init_g<<<grid, block>>>(d_g, d_rho, d_ux, d_uy, d_uz,
                                d_phi, d_force_realx, d_force_realy,
                                d_force_realz, d_ted);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaMemcpy(d_f_collide, d_f, NPOP * LXYZ * sizeof(double), cudaMemcpyDeviceToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_g_collide, d_g, NPOP * LXYZ * sizeof(double), cudaMemcpyDeviceToDevice));

        if (ACTIVATE_PIPE && NPIPE > 0) {
            build_pipe_links();
        }
    }
}

__global__ void init_f(double *f, double *rho, double *ux, double *uy, double *uz) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= LX || iy >= LY || iz >= LZ) return;

    int idx = (iz * LXY) + (iy * LX) + ix;

    const double r = 0.0;
    const double u = 0.0;
    const double v = 0.0;
    const double w = 0.0;
    const double RT = 1.0 / 3.0;
    const double uv = (u * u + v * v + w * w) / RT;

    rho[idx] = RHO0;
    ux[idx]  = u;
    uy[idx]  = v;
    uz[idx]  = w;

    for (int ip = 0; ip < NPOP; ip++) {
        const double eu = (d_cix[ip] * u + d_ciy[ip] * v + d_ciz[ip] * w) / RT;
        const double feq = d_tp[ip] * (r + eu + 0.5 * (eu * eu - uv));
        f[ip * LXYZ + idx] = feq;
    }
}

__global__ void init_g(double *g, double *rho, double *ux, double *uy, double *uz,
                       double *phi, double *force_realx, double *force_realy,
                       double *force_realz, double *ted) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= LX || iy >= LY || iz >= LZ) return;

    int idx = (iz * LXY) + (iy * LX) + ix;

    const double r = rho[idx];
    const double u = ux[idx];
    const double v = uy[idx];
    const double w = uz[idx];
    const double RT = 1.0 / 3.0;
    const double tempT = d_t0;
    const double uv = (u * u + v * v + w * w) / RT;
    const double lambda0 = 1.0 - d_tp[0];

    phi[idx] = tempT;
    force_realx[idx] = d_grav0 * d_beta * (tempT - d_t0);
    force_realy[idx] = 0.0;
    force_realz[idx] = 0.0;
    if (ted) ted[idx] = 0.0;

    for (int ip = 0; ip < NPOP; ip++) {
        const double eu = (d_cix[ip] * u + d_ciy[ip] * v + d_ciz[ip] * w) / RT;
        double geq = d_tp[ip] * tempT * (1.0 + eu + 0.5 * (eu * eu - uv));
        if (ip == 0) {
            geq -= lambda0 * tempT * r;
        } else {
            geq += d_tp[ip] * tempT * r;
        }
        g[ip * LXYZ + idx] = geq;
    }
}
