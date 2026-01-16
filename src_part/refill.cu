#include "particle.h"
#include <algorithm>

namespace {

__global__ void refill_new_fluid_kernel(const int *nodes,
                                        const int *owners,
                                        int count,
                                        double *f, double *g,
                                        const double *f_collide,
                                        const double *g_collide,
                                        const int *ibnode_prev,
                                        const int *owner_prev_map,
                                        const double *ppos_x, const double *ppos_y,
                                        const double *ppos_z,
                                        const double *pvel_x, const double *pvel_y,
                                        const double *pvel_z,
                                        const double *pomega_x, const double *pomega_y,
                                        const double *pomega_z,
                                        const double *ptemp,
                                        double *rho, double *ux, double *uy,
                                        double *uz,
                                        double *phi,
                                        double *force_x, double *force_y,
                                        double *force_z) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    const int idx = nodes[tid];
    const int iz = idx / LXY;
    const int iy = (idx - iz * LXY) / LX;
    const int ix = idx - iz * LXY - iy * LX;

    const double lambda0 = 1.0 - d_tp[0];
    const double RT = 1.0 / 3.0;

    int pid = owners ? owners[tid] : -1; 
    if (pid < 0 && owner_prev_map) {
        int prev_pid = owner_prev_map[idx];
        if (prev_pid >= 0) pid = prev_pid;
    }

    double rho_loc = 0.0;
    double mom_x = 0.0;
    double mom_y = 0.0;
    double mom_z = 0.0;
    double phi_loc = 0.0;

    for (int ip = 0; ip < NPOP; ++ip) {
        double temp_f = 0.0;
        double temp_g = 0.0;
        int sample_count = 0;


        for (int ip2 = 1; ip2 < NPOP; ++ip2) {
            int ix1 = ix + d_cix[ip2];
            int iy1 = iy + d_ciy[ip2];
            int iz1 = iz + d_ciz[ip2];
            if (ix1 < 0 || ix1 >= LX || iy1 < 0 || iy1 >= LY || iz1 < 0 || iz1 >= LZ) continue;
            int idx1 = iz1 * LXY + iy1 * LX + ix1;
            if (ibnode_prev[idx1] != 0) continue; 

            int ix2 = ix + 2 * d_cix[ip2];
            int iy2 = iy + 2 * d_ciy[ip2];
            int iz2 = iz + 2 * d_ciz[ip2];
            bool inside2 = (ix2 >= 0 && ix2 < LX && iy2 >= 0 && iy2 < LY && iz2 >= 0 && iz2 < LZ);
            if (inside2) { 
                int idx2 = iz2 * LXY + iy2 * LX + ix2;
                if (ibnode_prev[idx2] == 0) {
                    temp_f += 2.0 * f_collide[ip * LXYZ + idx1] - f_collide[ip * LXYZ + idx2];
                    temp_g += 2.0 * g_collide[ip * LXYZ + idx1] - g_collide[ip * LXYZ + idx2];
                    ++sample_count;
                    continue;
                }
            }

            temp_f += f_collide[ip * LXYZ + idx1];
            temp_g += g_collide[ip * LXYZ + idx1];
            ++sample_count;
        }

        if (sample_count == 0) {
            double px = 0.0;
            double py = 0.0;
            double pz = 0.0;
            double vx = 0.0;
            double vy = 0.0;
            double vz = 0.0;
            double omega_x = 0.0;
            double omega_y = 0.0;
            double omega_z = 0.0;
            double tempT = d_t0;

            if (pid >= 0) {
                px = ppos_x[pid];
                py = ppos_y[pid];
                pz = ppos_z[pid];
                vx = pvel_x[pid];
                vy = pvel_y[pid];
                vz = pvel_z[pid];
                omega_x = pomega_x[pid];
                omega_y = pomega_y[pid];
                omega_z = pomega_z[pid];
                tempT = ptemp[pid];
            }

            const double x = static_cast<double>(ix) + 0.5;
            const double y = static_cast<double>(iy) + 0.5;
            const double z = static_cast<double>(iz) + 0.5;
            const double relx = x - px;
            const double rely = y - py;
            const double relz = z - pz;
            double u = vx + (relz * omega_y - rely * omega_z);
            double v = vy + (relx * omega_z - relz * omega_x);
            double w = vz + (rely * omega_x - relx * omega_y);
            if (pid < 0) {
                u = 0.0;
                v = 0.0;
                w = 0.0;
            }

            const double r = RHO0;
            const double eu = d_cix[ip] * u + d_ciy[ip] * v + d_ciz[ip] * w;
            const double uv = (u * u + v * v + w * w) / RT;

            temp_f = d_tp[ip] * (r + eu + 0.5 * (eu * eu - uv));
            temp_g = d_tp[ip] * tempT * (1.0 + eu + 0.5 * (eu * eu - uv));
            if (ip == 0) {
                temp_g -= lambda0 * tempT * r;
            } else {
                temp_g += d_tp[ip] * tempT * r;
            }
            sample_count = 1;
        }

        const double inv_count = 1.0 / static_cast<double>(sample_count);
        const double f_val = temp_f * inv_count;
        const double g_val = temp_g * inv_count;

        f[ip * LXYZ + idx] = f_val;
        g[ip * LXYZ + idx] = g_val;

        rho_loc += f_val;
        mom_x += d_cix[ip] * f_val;
        mom_y += d_ciy[ip] * f_val;
        mom_z += d_ciz[ip] * f_val;
        phi_loc += g_val;
    }

    rho[idx] = rho_loc;
    if (rho_loc > 1e-12) {
        const double inv_rho = 1.0 / rho_loc;
        ux[idx] = mom_x * inv_rho;
        uy[idx] = mom_y * inv_rho;
        uz[idx] = mom_z * inv_rho;
    } else {
        ux[idx] = 0.0;
        uy[idx] = 0.0;
        uz[idx] = 0.0;
    }
    phi[idx] = phi_loc;

    if (force_x) force_x[idx] = 0.0;
    if (force_y) force_y[idx] = 0.0;
    if (force_z) force_z[idx] = 0.0;
}

} // namespace

void refill_nodes() {
    if (!ACTIVATE_PARTICLES || NPART == 0) return;

    if (num_new_fluid_nodes > 0 && d_new_fluid_nodes && d_new_fluid_pids) {
        const int threads = 128;
        const int blocks = (num_new_fluid_nodes + threads - 1) / threads;
        refill_new_fluid_kernel<<<blocks, threads>>>(d_new_fluid_nodes,
                                                     d_new_fluid_pids,
                                                     num_new_fluid_nodes,
                                                     d_f, d_g,
                                                     d_f_collide, d_g_collide,
                                                     d_ibnode_prev,
                                                     d_ibnode_owner_prev,
                                                     d_ppos_x, d_ppos_y, d_ppos_z,
                                                     d_pvel_x, d_pvel_y, d_pvel_z,
                                                     d_pomega_x, d_pomega_y, d_pomega_z,
                                                     d_ptemp,
                                                     d_rho, d_ux, d_uy, d_uz, d_phi,
                                                     d_force_realx, d_force_realy,
                                                     d_force_realz);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    CHECK_CUDA_ERROR(cudaMemcpy(d_ibnode_prev,  d_ibnode,  LXYZ * sizeof(int), cudaMemcpyDeviceToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_ibnode_owner_prev, d_ibnode_owner, LXYZ * sizeof(int), cudaMemcpyDeviceToDevice));
}
