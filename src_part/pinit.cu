#include "particle.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace {

void allocate_particle_arrays() {
    const size_t np = static_cast<size_t>(NPART);
    if (np == 0) return;

    h_ppos_x        = malloc_host_array<double>(np);
    h_ppos_y        = malloc_host_array<double>(np);
    h_ppos_z        = malloc_host_array<double>(np);
    h_pvel_x        = malloc_host_array<double>(np);
    h_pvel_y        = malloc_host_array<double>(np);
    h_pvel_z        = malloc_host_array<double>(np);
    h_pomega_x      = malloc_host_array<double>(np);
    h_pomega_y      = malloc_host_array<double>(np);
    h_pomega_z      = malloc_host_array<double>(np);
    h_ptheta_x      = malloc_host_array<double>(np);
    h_ptheta_y      = malloc_host_array<double>(np);
    h_ptheta_z      = malloc_host_array<double>(np);
    h_pforce_x      = malloc_host_array<double>(np);
    h_pforce_y      = malloc_host_array<double>(np);
    h_pforce_z      = malloc_host_array<double>(np);
    h_ptorque_x     = malloc_host_array<double>(np);
    h_ptorque_y     = malloc_host_array<double>(np);
    h_ptorque_z     = malloc_host_array<double>(np);
    h_pforce_hist_x = malloc_host_array<double>(np);
    h_pforce_hist_y = malloc_host_array<double>(np);
    h_pforce_hist_z = malloc_host_array<double>(np);
    h_ptorque_hist_x  = malloc_host_array<double>(np);
    h_ptorque_hist_y  = malloc_host_array<double>(np);
    h_ptorque_hist_z  = malloc_host_array<double>(np);
    h_ptemp         = malloc_host_array<double>(np);
    h_pheat         = malloc_host_array<double>(np);
    h_pheat_prev    = malloc_host_array<double>(np);

    CHECK_CUDA_ERROR(cudaMalloc(&d_ppos_x,        np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_ppos_y,        np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_ppos_z,        np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pvel_x,        np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pvel_y,        np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pvel_z,        np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pomega_x,      np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pomega_y,      np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pomega_z,      np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_ptheta_x,      np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_ptheta_y,      np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_ptheta_z,      np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pforce_x,      np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pforce_y,      np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pforce_z,      np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_ptorque_x,     np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_ptorque_y,     np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_ptorque_z,     np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pforce_hist_x, np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pforce_hist_y, np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pforce_hist_z, np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_ptorque_hist_x,  np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_ptorque_hist_y,  np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_ptorque_hist_z,  np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_ptemp,         np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pheat,         np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pheat_prev,    np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_particle_force_accum_x, np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_particle_force_accum_y, np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_particle_force_accum_z, np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_particle_torque_accum_x,  np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_particle_torque_accum_y,  np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_particle_torque_accum_z,  np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_particle_force_prev_x,  np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_particle_force_prev_y,  np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_particle_force_prev_z,  np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_particle_force_prev2_x, np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_particle_force_prev2_y, np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_particle_force_prev2_z, np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_particle_torque_prev_x,   np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_particle_torque_prev_y,   np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_particle_torque_prev_z,   np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_particle_torque_prev2_x,  np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_particle_torque_prev2_y,  np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_particle_torque_prev2_z,  np * sizeof(double)));

    // Grid markers & ownership (ibnode: 0 fluid, 1 solid)
    h_ibnode       = malloc_host_array<int>(LXYZ);   //0 res fluid, 1 res particle
    h_ibnode_prev  = malloc_host_array<int>(LXYZ);
    h_ibnode_owner = malloc_host_array<int>(LXYZ);   // current particle owner (-1 for fluid)
    h_ibnode_owner_prev = malloc_host_array<int>(LXYZ);
    CHECK_CUDA_ERROR(cudaMalloc(&d_ibnode,       LXYZ * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_ibnode_prev,  LXYZ * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_ibnode_owner, LXYZ * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_ibnode_owner_prev, LXYZ * sizeof(int)));
}

void generate_initial_positions() { //Ensure particles do not overlap
    if (NPART == 0) return;

    const double radius = particle_radius;

    std::srand(PARTICLE_RAND_SEED);
    const double min_pos_x = 1.5 * radius;
    const double min_pos_y = 1.5 * radius;
    const double min_pos_z = 1.5 * radius;
    const double span_x = static_cast<double>(LX) - 3.0 * radius;
    const double span_y = static_cast<double>(LY) - 3.0 * radius;
    const double span_z = static_cast<double>(LZ) - 3.0 * radius;
    long counter = 0;

    for (int p = 0; p < NPART; ++p) {
        while (true) {
            double rx = static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX);
            double ry = static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX);
            double rz = static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX);
            double x = min_pos_x + rx * span_x;
            double y = min_pos_y + ry * span_y;
            double z = min_pos_z + rz * span_z;

            bool ok = true;
            for (int q = 0; q < p; ++q) {
                double dx = x - h_ppos_x[q];
                double dy = y - h_ppos_y[q];
                double dz = z - h_ppos_z[q];
                double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (dist - 2.0 * radius < radius / 3.0) {
                    counter++;
                    if (counter > 1000000) {
                        fprintf(stderr, "Can't distribute the particles, iter. limit=%ld\n", counter);
                        exit(EXIT_FAILURE);
                    }
                    ok = false;
                    break;
                }
            }
            if (ok) {
                h_ppos_x[p] = x;
                h_ppos_y[p] = y;
                h_ppos_z[p] = z;
                break;
            }
        }
    }
}

void build_initial_ibnode() {
    const double radius_sq = particle_radius * particle_radius;
    for (int idx = 0; idx < LXYZ; ++idx) {
        h_ibnode[idx] = 0;  //0 is fuild; 1 is part
        h_ibnode_owner[idx] = -1; //particle ID
    }
    for (int iz = 0; iz < LZ; ++iz) {
        for (int iy = 0; iy < LY; ++iy) {
            for (int ix = 0; ix < LX; ++ix) {
                const double cx = static_cast<double>(ix) + 0.5;
                const double cy = static_cast<double>(iy) + 0.5;
                const double cz = static_cast<double>(iz) + 0.5;
                int idx = iz * LXY + iy * LX + ix;
                for (int p = 0; p < NPART; ++p) {
                    double dx = cx - h_ppos_x[p];
                    double dy = cy - h_ppos_y[p];
                    double dz = cz - h_ppos_z[p];
                    if (dx * dx + dy * dy + dz * dz <= radius_sq) {
                        h_ibnode[idx] = 1;
                        h_ibnode_owner[idx] = p; //assgin for particle ID
                        break;
                    }
                }
            }
        }
    }
    std::copy(h_ibnode, h_ibnode + LXYZ, h_ibnode_prev);
    std::copy(h_ibnode_owner, h_ibnode_owner + LXYZ, h_ibnode_owner_prev);

    CHECK_CUDA_ERROR(cudaMemcpy(d_ibnode,       h_ibnode,       LXYZ * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_ibnode_prev,  h_ibnode_prev,  LXYZ * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_ibnode_owner, h_ibnode_owner, LXYZ * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_ibnode_owner_prev, h_ibnode_owner_prev, LXYZ * sizeof(int), cudaMemcpyHostToDevice));
}

} // namespace

void init_particles() {
    init_particle_constants();

    if (!ACTIVATE_PARTICLES || NPART == 0) {
        // Ensure ibnode arrays exist even when ACTIVATE_PARTICLES is 0
        h_ibnode = malloc_host_array<int>(LXYZ);
        h_ibnode_prev = malloc_host_array<int>(LXYZ);
        h_ibnode_owner = malloc_host_array<int>(LXYZ);
        h_ibnode_owner_prev = malloc_host_array<int>(LXYZ);
        if (!h_ibnode || !h_ibnode_prev || !h_ibnode_owner || !h_ibnode_owner_prev) {
            fprintf(stderr, "Failed to allocate ibnode buffers.\n");
            exit(EXIT_FAILURE);
        }
        std::fill(h_ibnode, h_ibnode + LXYZ, 0);  // void fill(start_loc, end_loc, value)
        std::copy(h_ibnode, h_ibnode + LXYZ, h_ibnode_prev); // void copy(start_src, end_src, dst)
        std::fill(h_ibnode_owner, h_ibnode_owner + LXYZ, -1);
        std::copy(h_ibnode_owner, h_ibnode_owner + LXYZ, h_ibnode_owner_prev);
        CHECK_CUDA_ERROR(cudaMalloc(&d_ibnode,       LXYZ * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_ibnode_prev,  LXYZ * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_ibnode_owner, LXYZ * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_ibnode_owner_prev, LXYZ * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMemcpy(d_ibnode,       h_ibnode,       LXYZ * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_ibnode_prev,  h_ibnode_prev,  LXYZ * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_ibnode_owner, h_ibnode_owner, LXYZ * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_ibnode_owner_prev, h_ibnode_owner_prev, LXYZ * sizeof(int), cudaMemcpyHostToDevice));
        return;
    }

    allocate_particle_arrays();

    std::fill(h_pvel_x,        h_pvel_x        + NPART, 0.0);
    std::fill(h_pvel_y,        h_pvel_y        + NPART, 0.0);
    std::fill(h_pvel_z,        h_pvel_z        + NPART, 0.0);
    std::fill(h_pomega_x,      h_pomega_x      + NPART, 0.0);
    std::fill(h_pomega_y,      h_pomega_y      + NPART, 0.0);
    std::fill(h_pomega_z,      h_pomega_z      + NPART, 0.0);
    std::fill(h_ptheta_x,      h_ptheta_x      + NPART, 0.0);
    std::fill(h_ptheta_y,      h_ptheta_y      + NPART, 0.0);
    std::fill(h_ptheta_z,      h_ptheta_z      + NPART, 0.0);
    std::fill(h_pforce_x,      h_pforce_x      + NPART, 0.0);
    std::fill(h_pforce_y,      h_pforce_y      + NPART, 0.0);
    std::fill(h_pforce_z,      h_pforce_z      + NPART, 0.0);
    std::fill(h_ptorque_x,     h_ptorque_x     + NPART, 0.0);
    std::fill(h_ptorque_y,     h_ptorque_y     + NPART, 0.0);
    std::fill(h_ptorque_z,     h_ptorque_z     + NPART, 0.0);
    std::fill(h_pforce_hist_x, h_pforce_hist_x + NPART, 0.0);
    std::fill(h_pforce_hist_y, h_pforce_hist_y + NPART, 0.0);
    std::fill(h_pforce_hist_z, h_pforce_hist_z + NPART, 0.0);
    std::fill(h_ptorque_hist_x,  h_ptorque_hist_x  + NPART, 0.0);
    std::fill(h_ptorque_hist_y,  h_ptorque_hist_y  + NPART, 0.0);
    std::fill(h_ptorque_hist_z,  h_ptorque_hist_z  + NPART, 0.0);
    std::fill(h_pheat,         h_pheat         + NPART, 0.0);
    std::fill(h_pheat_prev,    h_pheat_prev    + NPART, 0.0);

    generate_initial_positions();

    const double initial_temp = PARTICLE_INIT_TEMP;
    std::fill(h_ptemp, h_ptemp + NPART, initial_temp);
    std::fill_n(h_particle_force_accum_x = malloc_host_array<double>(NPART), NPART, 0.0);  //void fill_n(start_loc, N , value)
    std::fill_n(h_particle_force_accum_y = malloc_host_array<double>(NPART), NPART, 0.0);
    std::fill_n(h_particle_force_accum_z = malloc_host_array<double>(NPART), NPART, 0.0);
    std::fill_n(h_particle_torque_accum_x  = malloc_host_array<double>(NPART), NPART, 0.0);
    std::fill_n(h_particle_torque_accum_y  = malloc_host_array<double>(NPART), NPART, 0.0);
    std::fill_n(h_particle_torque_accum_z  = malloc_host_array<double>(NPART), NPART, 0.0);

    CHECK_CUDA_ERROR(cudaMemcpy(d_ppos_x,        h_ppos_x,        NPART * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_ppos_y,        h_ppos_y,        NPART * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_ppos_z,        h_ppos_z,        NPART * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pvel_x,        h_pvel_x,        NPART * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pvel_y,        h_pvel_y,        NPART * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pvel_z,        h_pvel_z,        NPART * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pomega_x,      h_pomega_x,      NPART * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pomega_y,      h_pomega_y,      NPART * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pomega_z,      h_pomega_z,      NPART * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_ptheta_x,      h_ptheta_x,      NPART * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_ptheta_y,      h_ptheta_y,      NPART * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_ptheta_z,      h_ptheta_z,      NPART * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pforce_x,      h_pforce_x,      NPART * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pforce_y,      h_pforce_y,      NPART * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pforce_z,      h_pforce_z,      NPART * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_ptorque_x,     h_ptorque_x,     NPART * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_ptorque_y,     h_ptorque_y,     NPART * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_ptorque_z,     h_ptorque_z,     NPART * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pforce_hist_x, h_pforce_hist_x, NPART * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pforce_hist_y, h_pforce_hist_y, NPART * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pforce_hist_z, h_pforce_hist_z, NPART * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_ptorque_hist_x,  h_ptorque_hist_x,  NPART * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_ptorque_hist_y,  h_ptorque_hist_y,  NPART * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_ptorque_hist_z,  h_ptorque_hist_z,  NPART * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_ptemp,         h_ptemp,         NPART * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pheat,         h_pheat,         NPART * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pheat_prev,    h_pheat_prev,    NPART * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_particle_force_accum_x, h_particle_force_accum_x, NPART * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_particle_force_accum_y, h_particle_force_accum_y, NPART * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_particle_force_accum_z, h_particle_force_accum_z, NPART * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_particle_torque_accum_x,  h_particle_torque_accum_x,  NPART * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_particle_torque_accum_y,  h_particle_torque_accum_y,  NPART * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_particle_torque_accum_z,  h_particle_torque_accum_z,  NPART * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemset(d_particle_force_prev_x,   0, NPART * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemset(d_particle_force_prev_y,   0, NPART * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemset(d_particle_force_prev_z,   0, NPART * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemset(d_particle_force_prev2_x,  0, NPART * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemset(d_particle_force_prev2_y,  0, NPART * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemset(d_particle_force_prev2_z,  0, NPART * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemset(d_particle_torque_prev_x,    0, NPART * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemset(d_particle_torque_prev_y,    0, NPART * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemset(d_particle_torque_prev_z,    0, NPART * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemset(d_particle_torque_prev2_x,   0, NPART * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemset(d_particle_torque_prev2_y,   0, NPART * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemset(d_particle_torque_prev2_z,   0, NPART * sizeof(double)));

    build_initial_ibnode();
}
