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

void diag_flow(int istep) {
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

    CHECK_CUDA_ERROR(cudaMemcpy(h_ux,  d_ux,  LXYZ * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_uy,  d_uy,  LXYZ * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_uz,  d_uz,  LXYZ * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_phi, d_phi, LXYZ * sizeof(double), cudaMemcpyDeviceToHost));

    double umean = 0.0;
    double vmean = 0.0;
    double wmean = 0.0;
    double tmean = 0.0;
    overwrite_particle_nodes(nullptr, h_ux, h_uy, h_uz, h_phi, nullptr, nullptr);
    for (int i = 0; i < LXYZ; i++) {
        umean += h_ux[i];
        vmean += h_uy[i];
        wmean += h_uz[i];
        tmean += h_phi[i];
    }
    umean /= LXYZ;
    vmean /= LXYZ;
    wmean /= LXYZ;
    tmean /= LXYZ;

    printf("Step %-7d umean = %14.6e, vmean = %14.6e, wmean = %14.6e, tmean = %14.6e\n",
           istep, umean, vmean, wmean, tmean);
    fflush(stdout);

    char stat_filename[512];
    snprintf(stat_filename, sizeof(stat_filename), "%s/statistics.dat", dirname);
    FILE *f_stat = fopen(stat_filename, "a");
    fprintf(f_stat, "%-7d %14.6e %14.6e %14.6e %14.6e\n",
            istep, umean, vmean, wmean, tmean);
    fflush(f_stat);
    fclose(f_stat);
}

void output_nu(int istep) {
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

    char filename[512];
    snprintf(filename, sizeof(filename), "%s/Nu_%.2e.txt", dirname, rayl);

    CHECK_CUDA_ERROR(cudaMemcpy(h_ux,  d_ux,  LXYZ * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_phi, d_phi, LXYZ * sizeof(double), cudaMemcpyDeviceToHost));
    if (ACTIVATE_PARTICLES && NPART > 0 && h_ibnode) {
        CHECK_CUDA_ERROR(cudaMemcpy(h_ibnode, d_ibnode, LXYZ * sizeof(int), cudaMemcpyDeviceToHost));
    }

    const double volume_fraction = particle_fraction;
    const double fluid_fraction = fmax(0.0, 1.0 - volume_fraction);

    double plane_avg_uT_fluid = 0.0;
    double plane_avg_uT_part = 0.0;
    for (int ix = 0; ix < LX; ++ix) {
        double plane_sum_fluid = 0.0, plane_sum_part = 0.0;
        int fluid_count = 0, part_count = 0;
        for (int iz = 0; iz < LZ; ++iz) {
            for (int iy = 0; iy < LY; ++iy) {
                const int idx = iz * LXY + iy * LX + ix;
                bool is_fluid = true;
                if (ACTIVATE_PARTICLES && NPART > 0 && h_ibnode) {
                    is_fluid = (h_ibnode[idx] == 0);
                }
                if (is_fluid) {
                    plane_sum_fluid += h_ux[idx] * h_phi[idx];
                    fluid_count++;
                } else {
                    plane_sum_part += h_ux[idx] * h_phi[idx];
                    part_count++;
                }
            }
        }
        if (fluid_count > 0) {
            plane_avg_uT_fluid += plane_sum_fluid / static_cast<double>(fluid_count);
        }
        if (part_count > 0) {
            plane_avg_uT_part += plane_sum_part / static_cast<double>(part_count);
        }
    }
    plane_avg_uT_fluid /= static_cast<double>(LX);
    plane_avg_uT_part /= static_cast<double>(LX);

    const double conductive_flux = diff * (tHot - tCold) / static_cast<double>(LX);
    double Nu_bulk_fluid = fluid_fraction, Nu_bulk_part = 0.0;
    if (conductive_flux != 0.0) {
        Nu_bulk_fluid += fluid_fraction * (plane_avg_uT_fluid / conductive_flux);
        Nu_bulk_part = volume_fraction * CP_PARTICLE_RATIO * (plane_avg_uT_part / conductive_flux);
    }

    double Nu_bulk = Nu_bulk_fluid + Nu_bulk_part;

    double bottom = 0.0;
    double top = 0.0;

    int bottom_count = 0;
    int top_count = 0;
    for (int iz = 0; iz < LZ; ++iz) {
        for (int iy = 0; iy < LY; ++iy) {
            const int idx_bottom = iz * LXY + iy * LX;
            bool bottom_fluid = true;
            if (ACTIVATE_PARTICLES && NPART > 0 && h_ibnode) {
                bottom_fluid = (h_ibnode[idx_bottom] == 0);
            }
            if (bottom_fluid) {
                bottom += (tHot - h_phi[idx_bottom]) / 0.5;
                bottom_count++;
            }

            const int idx_top = iz * LXY + iy * LX + (LX - 1);
            bool top_fluid = true;
            if (ACTIVATE_PARTICLES && NPART > 0 && h_ibnode) {
                top_fluid = (h_ibnode[idx_top] == 0);
            }
            if (top_fluid) {
                top += (h_phi[idx_top] - tCold) / 0.5;
                top_count++;
            }
        }
    }

    const double section_area = static_cast<double>(LY) * static_cast<double>(LZ);
    const double wall_scale = static_cast<double>(LX) / (tHot - tCold);
    double Tbottom = 0.0;
    double Ttop = 0.0;

    if (bottom_count > 0) {
        Tbottom = (bottom / section_area) * wall_scale;
    }
    if (top_count > 0) {
        Ttop = (top / section_area) * wall_scale;
    }

    FILE *f_nu = fopen(filename, "a");
    fprintf(f_nu, "%15.6E%15.6E%15.6E%15.6E%15.6E%15.6E%15.6E\n",
            (double)istep, volume_fraction, Nu_bulk, Ttop, Tbottom, Nu_bulk_fluid, Nu_bulk_part);
    //                                      Nu_bulk  Nu_top Nu_bottom                                       
    fflush(f_nu);
    fclose(f_nu);
}

void output_profile(int istep) {
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

    char filename[512];
    snprintf(filename, sizeof(filename), "%s/Profile_%.2e.txt", dirname, rayl);

    CHECK_CUDA_ERROR(cudaMemcpy(h_rho, d_rho, LXYZ * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_ux,  d_ux,  LXYZ * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_uy,  d_uy,  LXYZ * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_uz,  d_uz,  LXYZ * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_phi, d_phi, LXYZ * sizeof(double), cudaMemcpyDeviceToHost));
    overwrite_particle_nodes(h_rho, h_ux, h_uy, h_uz, h_phi, nullptr, nullptr);

    FILE *f_profile = fopen(filename, "a");
    fprintf(f_profile, "%d\n", istep);

    for (int ix = 0; ix < LX; ix++) {
        double sum_ux = 0.0;
        double sum_uy = 0.0;
        double sum_uz = 0.0;
        double sum_phi = 0.0;
        double sum_phi_sq = 0.0;
        int count = 0;

        for (int iz = 0; iz < LZ; iz++) {
            for (int iy = 0; iy < LY; iy++) {
                int idx = iz * LXY + iy * LX + ix;
                sum_ux     += h_ux[idx];
                sum_uy     += h_uy[idx];
                sum_uz     += h_uz[idx];
                sum_phi    += h_phi[idx];
                sum_phi_sq += h_phi[idx] * h_phi[idx];
                count++;
            }
        }

        double inv_count = (count > 0) ? (1.0 / static_cast<double>(count)) : 0.0;
        double avg_ux  = sum_ux * inv_count;
        double avg_uy  = sum_uy * inv_count;
        double avg_uz  = sum_uz * inv_count;
        double avg_phi = sum_phi * inv_count;
        double phi_rms = sqrt(sum_phi_sq * inv_count - avg_phi * avg_phi);

        fprintf(f_profile, "%14.6e %14.6e %14.6e %14.6e\n",
                avg_ux, avg_uy, avg_uz, avg_phi);
        (void)phi_rms;
    }

    fflush(f_profile);
    fclose(f_profile);
}
