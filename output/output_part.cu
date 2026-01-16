#include "particle.h"
#include <vector>

void output_particles(int istep) {
    if (!ACTIVATE_PARTICLES || NPART == 0) return;

    char dirname[256];
    snprintf(dirname, sizeof(dirname), "Ra%.1ePr%.2f", rayl, prand);

    struct stat st = {0};
    if (stat(dirname, &st) == -1) {
#ifdef _WIN32
        _mkdir(dirname);
#else
        mkdir(dirname, 0700);
#endif
    }

    char posdir[1024];
    snprintf(posdir, sizeof(posdir), "%s/pos", dirname);
    if (stat(posdir, &st) == -1) {
#ifdef _WIN32
        _mkdir(posdir);
#else
        mkdir(posdir, 0700);
#endif
    }

    CHECK_CUDA_ERROR(cudaMemcpy(h_ppos_x, d_ppos_x, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_ppos_y, d_ppos_y, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_ppos_z, d_ppos_z, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_pvel_x, d_pvel_x, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_pvel_y, d_pvel_y, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_pvel_z, d_pvel_z, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_pforce_x, d_particle_force_accum_x, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_pforce_y, d_particle_force_accum_y, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_pforce_z, d_particle_force_accum_z, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_pomega_x, d_pomega_x, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_pomega_y, d_pomega_y, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_pomega_z, d_pomega_z, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_ptheta_x, d_ptheta_x, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_ptheta_y, d_ptheta_y, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_ptheta_z, d_ptheta_z, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_ptorque_x, d_particle_torque_accum_x, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_ptorque_y, d_particle_torque_accum_y, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_ptorque_z, d_particle_torque_accum_z, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_ptemp, d_ptemp, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_pheat, d_pheat, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_pheat_prev, d_pheat_prev, NPART * sizeof(double), cudaMemcpyDeviceToHost));

    std::vector<double> force_prev_x(NPART), force_prev_y(NPART), force_prev_z(NPART);
    std::vector<double> force_prev2_x(NPART), force_prev2_y(NPART), force_prev2_z(NPART);
    std::vector<double> torque_prev_x(NPART), torque_prev_y(NPART), torque_prev_z(NPART);
    std::vector<double> torque_prev2_x(NPART), torque_prev2_y(NPART), torque_prev2_z(NPART);

    CHECK_CUDA_ERROR(cudaMemcpy(force_prev_x.data(), d_particle_force_prev_x, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(force_prev_y.data(), d_particle_force_prev_y, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(force_prev_z.data(), d_particle_force_prev_z, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(force_prev2_x.data(), d_particle_force_prev2_x, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(force_prev2_y.data(), d_particle_force_prev2_y, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(force_prev2_z.data(), d_particle_force_prev2_z, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(torque_prev_x.data(), d_particle_torque_prev_x, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(torque_prev_y.data(), d_particle_torque_prev_y, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(torque_prev_z.data(), d_particle_torque_prev_z, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(torque_prev2_x.data(), d_particle_torque_prev2_x, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(torque_prev2_y.data(), d_particle_torque_prev2_y, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(torque_prev2_z.data(), d_particle_torque_prev2_z, NPART * sizeof(double), cudaMemcpyDeviceToHost));

    char filename[2048];
    int n = snprintf(filename, sizeof(filename), "%s/pos_%09d.txt", posdir, istep);
    if (n < 0 || n >= (int)sizeof(filename)) {
        fprintf(stderr, "Failed to format particle output path (buffer too small)\n");
        return;
    }
    FILE *f_pos = fopen(filename, "w");
    if (!f_pos) {
        fprintf(stderr, "Failed to open %s for particle output\n", filename);
        return;
    }

    for (int pid = 0; pid < NPART; ++pid) {
        const double theta_x = h_ptheta_x[pid];
        const double theta_y = h_ptheta_y[pid];
        const double theta_z = h_ptheta_z[pid];
        const double omega_x = h_pomega_x[pid];
        const double omega_y = h_pomega_y[pid];
        const double omega_z = h_pomega_z[pid];
        const double torque_x = h_ptorque_x[pid];
        const double torque_y = h_ptorque_y[pid];
        const double torque_z = h_ptorque_z[pid];
        fprintf(f_pos,
                "%d "
                "%14.6e %14.6e %14.6e "
                "%14.6e %14.6e %14.6e "
                "%14.6e %14.6e %14.6e "
                "%14.6e %14.6e %14.6e "
                "%14.6e %14.6e %14.6e "
                "%14.6e %14.6e %14.6e "
                "%14.6e %14.6e %14.6e "
                "%14.6e %14.6e %14.6e "
                "%14.6e %14.6e %14.6e "
                "%14.6e %14.6e %14.6e "
                "%14.6e %14.6e %14.6e\n",
                pid,
                h_ppos_x[pid], h_ppos_y[pid], h_ppos_z[pid],
                h_pvel_x[pid], h_pvel_y[pid], h_pvel_z[pid],
                h_pforce_x[pid], h_pforce_y[pid], h_pforce_z[pid],
                force_prev_x[pid], force_prev_y[pid], force_prev_z[pid],
                force_prev2_x[pid], force_prev2_y[pid], force_prev2_z[pid],
                theta_x, theta_y, theta_z,
                omega_x, omega_y, omega_z,
                torque_x, torque_y, torque_z,
                torque_prev_x[pid], torque_prev_y[pid], torque_prev_z[pid],
                torque_prev2_x[pid], torque_prev2_y[pid], torque_prev2_z[pid],
                h_ptemp[pid], h_pheat[pid], h_pheat_prev[pid]);
    }

    fclose(f_pos);
}
