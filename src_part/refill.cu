#include "particle.h"
#include <algorithm>
#include <cmath>
#include <cub/cub.cuh>

namespace {

__global__ void new_fluid_count_kernel(int *count,
                                       const int *ibnode_prev, const int *ibnode,
                                       const int *owner_prev,
                                       const double *ppos_x,
                                       const double *ppos_y,
                                       const double *ppos_z) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= LXYZ) return;

    if (ibnode_prev[idx] != 1 || ibnode[idx] == 1) {
        count[idx] = 0;
        return;
    }

    int pid = owner_prev[idx];
    if (pid < 0 || pid >= NPART) {
        count[idx] = 0;
        return;
    }

    const int iz = idx / LXY;
    const int rem = idx - iz * LXY;
    const int iy = rem / LX;
    const int ix = rem - iy * LX;
    const double x0 = static_cast<double>(ix) + 0.5;
    const double y0 = static_cast<double>(iy) + 0.5;
    const double z0 = static_cast<double>(iz) + 0.5;
    const double dx = x0 - ppos_x[pid];
    const double dy = y0 - ppos_y[pid];
    const double dz = z0 - ppos_z[pid];
    const double dist = sqrt(dx * dx + dy * dy + dz * dz);
    const double close_thresh = 1.7606816861659007;
    count[idx] = (dist - d_particle_radius <= close_thresh) ? 1 : 0;
}

__global__ void new_fluid_fill_kernel(const int *count, const int *offset,
                                      const int *ibnode_prev, const int *ibnode,
                                      const int *owner_prev,
                                      const double *ppos_x,
                                      const double *ppos_y,
                                      const double *ppos_z,
                                      int *nodes, int *pids) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= LXYZ) return;
    if (count[idx] == 0) return;

    if (ibnode_prev[idx] != 1 || ibnode[idx] == 1) return;

    int pid = owner_prev[idx];
    if (pid < 0 || pid >= NPART) return;

    const int iz = idx / LXY;
    const int rem = idx - iz * LXY;
    const int iy = rem / LX;
    const int ix = rem - iy * LX;
    const double x0 = static_cast<double>(ix) + 0.5;
    const double y0 = static_cast<double>(iy) + 0.5;
    const double z0 = static_cast<double>(iz) + 0.5;
    const double dx = x0 - ppos_x[pid];
    const double dy = y0 - ppos_y[pid];
    const double dz = z0 - ppos_z[pid];
    const double dist = sqrt(dx * dx + dy * dy + dz * dz);
    const double close_thresh = 1.7606816861659007;
    if (dist - d_particle_radius > close_thresh) return;

    const int pos = offset[idx];
    nodes[pos] = idx;
    pids[pos] = pid;
}

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

    int pid = owners ? owners[tid] : -1; //pid: 该新增点上一step所属的颗粒
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

        //如果该ip方向有相邻流体点，则使用相邻流体点插值构造f/g
        for (int ip2 = 1; ip2 < NPOP; ++ip2) {
            int ix1 = ix + d_cix[ip2];
            int iy1 = iy + d_ciy[ip2];
            int iz1 = iz + d_ciz[ip2];
            if (ix1 < 0 || ix1 >= LX || iy1 < 0 || iy1 >= LY || iz1 < 0 || iz1 >= LZ) continue;
            int idx1 = iz1 * LXY + iy1 * LX + ix1;
            if (ibnode_prev[idx1] != 0) continue; //相邻流体点上一时刻也必须为流体点（否则它也没有f/g）

            int ix2 = ix + 2 * d_cix[ip2];
            int iy2 = iy + 2 * d_ciy[ip2];
            int iz2 = iz + 2 * d_ciz[ip2];
            bool inside2 = (ix2 >= 0 && ix2 < LX && iy2 >= 0 && iy2 < LY && iz2 >= 0 && iz2 < LZ);
            if (inside2) { //1、2阶都有效
                int idx2 = iz2 * LXY + iy2 * LX + ix2;
                if (ibnode_prev[idx2] == 0) {
                    temp_f += 2.0 * f_collide[ip * LXYZ + idx1] - f_collide[ip * LXYZ + idx2];
                    temp_g += 2.0 * g_collide[ip * LXYZ + idx1] - g_collide[ip * LXYZ + idx2];
                    ++sample_count;
                    continue;
                }
            }

            //仅1阶有效
            temp_f += f_collide[ip * LXYZ + idx1];
            temp_g += g_collide[ip * LXYZ + idx1];
            ++sample_count;
        }

        //如果没有，则使用之前所属颗粒中心的宏观量->近似计算当前节点宏观量->使用宏观量来生成f/g
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

    //prepare new fluid nodes
    if (!d_link_count) {
        CHECK_CUDA_ERROR(cudaMalloc(&d_link_count, LXYZ * sizeof(int)));
    }
    if (!d_link_offset) {
        CHECK_CUDA_ERROR(cudaMalloc(&d_link_offset, LXYZ * sizeof(int)));
    }

    size_t scan_bytes = 0;
    CHECK_CUDA_ERROR(cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes,
                                                   d_link_count, d_link_offset, LXYZ));
    if (scan_bytes > d_link_scan_bytes) {
        if (d_link_scan_tmp) {
            CHECK_CUDA_ERROR(cudaFree(d_link_scan_tmp));
        }
        CHECK_CUDA_ERROR(cudaMalloc(&d_link_scan_tmp, scan_bytes));
        d_link_scan_bytes = scan_bytes;
    }
    const int threads = 256;
    const int blocks = (LXYZ + threads - 1) / threads;
    new_fluid_count_kernel<<<blocks, threads>>>(d_link_count,
                                                d_ibnode_prev, d_ibnode,
                                                d_ibnode_owner_prev,
                                                d_ppos_x, d_ppos_y, d_ppos_z);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cub::DeviceScan::ExclusiveSum(d_link_scan_tmp, d_link_scan_bytes,
                                                   d_link_count, d_link_offset, LXYZ));

    int last_offset = 0;
    int last_count = 0;
    CHECK_CUDA_ERROR(cudaMemcpy(&last_offset, d_link_offset + (LXYZ - 1),
                                sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(&last_count, d_link_count + (LXYZ - 1),
                                sizeof(int), cudaMemcpyDeviceToHost));
    num_new_fluid_nodes = last_offset + last_count;

    if (num_new_fluid_nodes > new_fluid_capacity) {
        if (d_new_fluid_nodes) {
            CHECK_CUDA_ERROR(cudaFree(d_new_fluid_nodes));
        }
        if (d_new_fluid_pids) {
            CHECK_CUDA_ERROR(cudaFree(d_new_fluid_pids));
        }
        CHECK_CUDA_ERROR(cudaMalloc(&d_new_fluid_nodes, num_new_fluid_nodes * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_new_fluid_pids, num_new_fluid_nodes * sizeof(int)));
        new_fluid_capacity = num_new_fluid_nodes;
    }

    if (num_new_fluid_nodes > 0) { //compress to array
        new_fluid_fill_kernel<<<blocks, threads>>>(d_link_count, d_link_offset,
                                                   d_ibnode_prev, d_ibnode,
                                                   d_ibnode_owner_prev,
                                                   d_ppos_x, d_ppos_y, d_ppos_z,
                                                   d_new_fluid_nodes, d_new_fluid_pids);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }

    //refill the new fluid nodes
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

    // 更新上一时刻缓存，供下一时间步使用
    CHECK_CUDA_ERROR(cudaMemcpy(d_ibnode_prev,  d_ibnode,  LXYZ * sizeof(int), cudaMemcpyDeviceToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_ibnode_owner_prev, d_ibnode_owner, LXYZ * sizeof(int), cudaMemcpyDeviceToDevice));
}
