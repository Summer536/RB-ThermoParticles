#include "particle.h"

namespace {

void overwrite_particle_nodes(double *rho, double *ux, double *uy, double *uz,
                              double *phi, double *ted, double *Ked) {
    if (!(ACTIVATE_PARTICLES && NPART > 0)) {
        return;
    }
    if (!h_ibnode || !h_ibnode_owner) {
        return;
    }

    CHECK_CUDA_ERROR(cudaMemcpy(h_ibnode, d_ibnode, LXYZ * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_ibnode_owner, d_ibnode_owner, LXYZ * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_ppos_x, d_ppos_x, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_ppos_y, d_ppos_y, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_ppos_z, d_ppos_z, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_pvel_x, d_pvel_x, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_pvel_y, d_pvel_y, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_pvel_z, d_pvel_z, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_pomega_x, d_pomega_x, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_pomega_y, d_pomega_y, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_pomega_z, d_pomega_z, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_ptemp,  d_ptemp,  NPART * sizeof(double), cudaMemcpyDeviceToHost));

    for (int idx = 0; idx < LXYZ; ++idx) {
        if (h_ibnode[idx] == 0) continue;

        int pid = h_ibnode_owner[idx];
        if (pid < 0 || pid >= NPART) continue;

        int iz = idx / LXY;
        int iy = (idx - iz * LXY) / LX;
        int ix = idx - iz * LXY - iy * LX;
        double cx = static_cast<double>(ix) + 0.5;
        double cy = static_cast<double>(iy) + 0.5;
        double cz = static_cast<double>(iz) + 0.5;

        double px = h_ppos_x[pid];
        double py = h_ppos_y[pid];
        double pz = h_ppos_z[pid];
        double relx = cx - px;
        double rely = cy - py;
        double relz = cz - pz;

        double vx = h_pvel_x[pid] + (relz * h_pomega_y[pid] - rely * h_pomega_z[pid]);
        double vy = h_pvel_y[pid] + (relx * h_pomega_z[pid] - relz * h_pomega_x[pid]);
        double vz = h_pvel_z[pid] + (rely * h_pomega_x[pid] - relx * h_pomega_y[pid]);
        double temp = h_ptemp[pid];

        if (rho) rho[idx] = RHO0;
        if (ux)  ux[idx]  = vx;
        if (uy)  uy[idx]  = vy;
        if (uz)  uz[idx]  = vz;
        if (phi) phi[idx] = temp;
        if (ted) ted[idx] = 0.0;
        if (Ked) Ked[idx] = 0.0;
    }
}

} // namespace

void output_flow(int istep) {
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

    dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 grid((LX + BLOCK_X - 1) / BLOCK_X,
              (LY + BLOCK_Y - 1) / BLOCK_Y,
              (LZ + BLOCK_Z - 1) / BLOCK_Z);
    compute_Ked<<<grid, block>>>(d_f_collide, d_force_realx, d_force_realy, d_force_realz,
                                 d_rho, d_ux, d_uy, d_uz, d_Ked);
    CHECK_CUDA_ERROR(cudaGetLastError());
    compute_ted<<<grid, block>>>(d_g_collide, d_force_realx, d_force_realy, d_force_realz,
                                 d_rho, d_ux, d_uy, d_uz, d_phi, d_ted);
    CHECK_CUDA_ERROR(cudaGetLastError());

    CHECK_CUDA_ERROR(cudaMemcpy(h_rho, d_rho, LXYZ * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_ux,  d_ux,  LXYZ * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_uy,  d_uy,  LXYZ * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_uz,  d_uz,  LXYZ * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_phi, d_phi, LXYZ * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_ted, d_ted, LXYZ * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_Ked, d_Ked, LXYZ * sizeof(double), cudaMemcpyDeviceToHost));
    overwrite_particle_nodes(h_rho, h_ux, h_uy, h_uz, h_phi, h_ted, h_Ked);

    char data_filename[512];
    snprintf(data_filename, sizeof(data_filename), "%s/%09d.dat", dirname, istep);

    FILE *f_data = fopen(data_filename, "wb");
    fwrite(h_rho, sizeof(double), LXYZ, f_data);
    fwrite(h_ux,  sizeof(double), LXYZ, f_data);
    fwrite(h_uy,  sizeof(double), LXYZ, f_data);
    fwrite(h_uz,  sizeof(double), LXYZ, f_data);
    fwrite(h_phi, sizeof(double), LXYZ, f_data);
    fwrite(h_ted, sizeof(double), LXYZ, f_data);
    fwrite(h_Ked, sizeof(double), LXYZ, f_data);
    fflush(f_data);
    fclose(f_data);
}

void output_fg(int istep) {
    char dirname[256];
    snprintf(dirname, sizeof(dirname), "Ra%.1ePr%.2f", rayl, prand);

    char fg_filename[512];
    snprintf(fg_filename, sizeof(fg_filename), "%s/fg%09d.dat", dirname, istep);

    CHECK_CUDA_ERROR(cudaMemcpy(h_f, d_f, NPOP * LXYZ * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_g, d_g, NPOP * LXYZ * sizeof(double), cudaMemcpyDeviceToHost));

    FILE *f_fg = fopen(fg_filename, "wb");
    fwrite(h_f, sizeof(double), NPOP * LXYZ, f_fg);
    fwrite(h_g, sizeof(double), NPOP * LXYZ, f_fg);
    fflush(f_fg);
    fclose(f_fg);
}
