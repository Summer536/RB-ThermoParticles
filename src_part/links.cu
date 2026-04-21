#include "particle.h"
#include <cmath>
#include <cub/cub.cuh>

namespace {

__device__ inline bool compute_link_q(int ip, double x0, double y0, double z0,
                                      double px, double py, double pz,
                                      double radius_sq, double &q) {
    const double cx_dir = static_cast<double>(d_cix[ip]);
    const double cy_dir = static_cast<double>(d_ciy[ip]);
    const double cz_dir = static_cast<double>(d_ciz[ip]);
    const double dx = x0 - px;
    const double dy = y0 - py;
    const double dz = z0 - pz;
    const double a = cx_dir * cx_dir + cy_dir * cy_dir + cz_dir * cz_dir;
    const double b = 2.0 * (cx_dir * dx + cy_dir * dy + cz_dir * dz);
    const double c = dx * dx + dy * dy + dz * dz - radius_sq;
    const double disc = b * b - 4.0 * a * c;
    if (disc <= 0.0) return false;
    q = (-b - sqrt(disc)) / (2.0 * a);
    return (q >= 0.0 && q <= 1.0);
}

__global__ void mark_ibnode_kernel(int pid,
                                   int ix_min, int iy_min, int iz_min,
                                   int ix_max, int iy_max, int iz_max,
                                   double px, double py, double pz,
                                   double radius_sq,
                                   int *ibnode, int *owner) {
    int ix = ix_min + blockIdx.x * blockDim.x + threadIdx.x;
    int iy = iy_min + blockIdx.y * blockDim.y + threadIdx.y;
    int iz = iz_min + blockIdx.z * blockDim.z + threadIdx.z;
    if (ix > ix_max || iy > iy_max || iz > iz_max) return;

    const double cx = static_cast<double>(ix) + 0.5;
    const double cy = static_cast<double>(iy) + 0.5;
    const double cz = static_cast<double>(iz) + 0.5;
    const double dx = cx - px;
    const double dy = cy - py;
    const double dz = cz - pz;
    if (dx * dx + dy * dy + dz * dz > radius_sq) return;

    const int idx = iz * LXY + iy * LX + ix;
    if (owner[idx] != -1) return;
    owner[idx] = pid;
    ibnode[idx] = 1;
}

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
    const double close_thresh = 1.7606816861659007;  //sqrt(3.1)
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
    const double close_thresh = 1.7606816861659007;  //sqrt(3.1)
    if (dist - d_particle_radius > close_thresh) return;

    const int pos = offset[idx];
    nodes[pos] = idx;
    pids[pos] = pid;
}

__global__ void link_count_kernel(int *count,
                                  const int *ibnode,
                                  const int *owner,
                                  const double *ppos_x,
                                  const double *ppos_y,
                                  const double *ppos_z) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= LXYZ) return;
    if (ibnode[idx] == 1) {
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
    const double radius_sq = d_particle_radius * d_particle_radius;

    int cnt = 0;
    for (int ip = 1; ip < NPOP; ++ip) {
        const int nx = ix + d_cix[ip];
        const int ny = iy + d_ciy[ip];
        const int nz = iz + d_ciz[ip];
        if (nx < 0 || nx >= LX || ny < 0 || ny >= LY || nz < 0 || nz >= LZ) continue;
        const int nidx = nz * LXY + ny * LX + nx;
        const int pid = owner[nidx];
        if (pid < 0) continue;
        double q = 0.0;
        if (compute_link_q(ip, x0, y0, z0, ppos_x[pid], ppos_y[pid], ppos_z[pid], radius_sq, q)) {
            cnt++;
        }
    }
    count[idx] = cnt;
}

__global__ void link_fill_kernel(const int *count, const int *offset,
                                 const int *ibnode,
                                 const int *owner,
                                 const double *ppos_x,
                                 const double *ppos_y,
                                 const double *ppos_z,
                                 ParticleLink *links) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= LXYZ) return;
    if (count[idx] == 0 || ibnode[idx] == 1) return;

    const int iz = idx / LXY;
    const int rem = idx - iz * LXY;
    const int iy = rem / LX;
    const int ix = rem - iy * LX;
    const double x0 = static_cast<double>(ix) + 0.5;
    const double y0 = static_cast<double>(iy) + 0.5;
    const double z0 = static_cast<double>(iz) + 0.5;
    const double radius_sq = d_particle_radius * d_particle_radius;

    int write_idx = offset[idx];
    for (int ip = 1; ip < NPOP; ++ip) {
        const int nx = ix + d_cix[ip];
        const int ny = iy + d_ciy[ip];
        const int nz = iz + d_ciz[ip];
        if (nx < 0 || nx >= LX || ny < 0 || ny >= LY || nz < 0 || nz >= LZ) continue;
        const int nidx = nz * LXY + ny * LX + nx;
        const int pid = owner[nidx];
        if (pid < 0) continue;

        double q = 0.0;
        if (!compute_link_q(ip, x0, y0, z0, ppos_x[pid], ppos_y[pid], ppos_z[pid], radius_sq, q)) {
            continue;
        }

        const double cx_dir = static_cast<double>(d_cix[ip]);
        const double cy_dir = static_cast<double>(d_ciy[ip]);
        const double cz_dir = static_cast<double>(d_ciz[ip]);
        ParticleLink link;
        link.cell_i = ix;
        link.cell_j = iy;
        link.cell_k = iz;
        link.dir = ip;
        link.pid = pid;
        link.q = q;
        link.rx = q * cx_dir;
        link.ry = q * cy_dir;
        link.rz = q * cz_dir;
        links[write_idx++] = link;
    }
}

} // namespace

void build_links() {
    if (!ACTIVATE_PARTICLES || NPART == 0) return;
    // Refresh particle positions from device to host for bounding boxes.
    CHECK_CUDA_ERROR(cudaMemcpy(h_ppos_x, d_ppos_x, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_ppos_y, d_ppos_y, NPART * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_ppos_z, d_ppos_z, NPART * sizeof(double), cudaMemcpyDeviceToHost));

    // Preserve previous ibnode map for state transition detection.
    CHECK_CUDA_ERROR(cudaMemcpy(d_ibnode_prev, d_ibnode, LXYZ * sizeof(int), cudaMemcpyDeviceToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_ibnode_owner_prev, d_ibnode_owner, LXYZ * sizeof(int), cudaMemcpyDeviceToDevice));

    // Clear current ibnode and owner.
    CHECK_CUDA_ERROR(cudaMemset(d_ibnode, 0, LXYZ * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemset(d_ibnode_owner, 0xFF, LXYZ * sizeof(int)));

    const double radius_sq = particle_radius * particle_radius;
    const dim3 block_ib(8, 8, 8);


    for (int p = 0; p < NPART; ++p) {
        const double px = h_ppos_x[p];
        const double py = h_ppos_y[p];
        const double pz = h_ppos_z[p];
        int ix_min = static_cast<int>(floor(px - particle_radius - 0.5));  // local search
        int ix_max = static_cast<int>(floor(px + particle_radius - 0.5));  // local search
        int iy_min = static_cast<int>(floor(py - particle_radius - 0.5));  // local search
        int iy_max = static_cast<int>(floor(py + particle_radius - 0.5));  // local search
        int iz_min = static_cast<int>(floor(pz - particle_radius - 0.5));  // local search
        int iz_max = static_cast<int>(floor(pz + particle_radius - 0.5));  // local search

        if (ix_min < 0) ix_min = 0;
        if (iy_min < 0) iy_min = 0;
        if (iz_min < 0) iz_min = 0;
        if (ix_max > LX - 1) ix_max = LX - 1;
        if (iy_max > LY - 1) iy_max = LY - 1;
        if (iz_max > LZ - 1) iz_max = LZ - 1;
        if (ix_min > ix_max || iy_min > iy_max || iz_min > iz_max) continue;

        const dim3 grid_ib((ix_max - ix_min + block_ib.x) / block_ib.x,
                           (iy_max - iy_min + block_ib.y) / block_ib.y,
                           (iz_max - iz_min + block_ib.z) / block_ib.z);
        mark_ibnode_kernel<<<grid_ib, block_ib>>>(p,
                                                  ix_min, iy_min, iz_min,
                                                  ix_max, iy_max, iz_max,
                                                  px, py, pz,
                                                  radius_sq,
                                                  d_ibnode, d_ibnode_owner);
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

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

//////////////////////////////////////////////////////////
    // New fluid nodes.
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
    if (num_new_fluid_nodes > 0) {
        new_fluid_fill_kernel<<<blocks, threads>>>(d_link_count, d_link_offset,
                                                   d_ibnode_prev, d_ibnode,
                                                   d_ibnode_owner_prev,
                                                   d_ppos_x, d_ppos_y, d_ppos_z,
                                                   d_new_fluid_nodes, d_new_fluid_pids);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }
///////////////////////////////////////////////////////////////////////////

    // Link counting.
    link_count_kernel<<<blocks, threads>>>(d_link_count,
                                           d_ibnode, d_ibnode_owner,
                                           d_ppos_x, d_ppos_y, d_ppos_z);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cub::DeviceScan::ExclusiveSum(d_link_scan_tmp, d_link_scan_bytes,
                                                   d_link_count, d_link_offset, LXYZ));



    CHECK_CUDA_ERROR(cudaMemcpy(&last_offset, d_link_offset + (LXYZ - 1),
                                sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(&last_count, d_link_count + (LXYZ - 1),
                                sizeof(int), cudaMemcpyDeviceToHost));
    num_particle_links = last_offset + last_count;

    if (num_particle_links > particle_link_capacity) {
        if (d_particle_links) {
            CHECK_CUDA_ERROR(cudaFree(d_particle_links));
        }
        CHECK_CUDA_ERROR(cudaMalloc(&d_particle_links,
                                    num_particle_links * sizeof(ParticleLink)));
        particle_link_capacity = num_particle_links;
    }
    if (num_particle_links > 0) {
        link_fill_kernel<<<blocks, threads>>>(d_link_count, d_link_offset,
                                              d_ibnode, d_ibnode_owner,
                                              d_ppos_x, d_ppos_y, d_ppos_z,
                                              d_particle_links);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
