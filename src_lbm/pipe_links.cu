#include "pipe.h"  // @pinit.cu @globals.cu @links.cu @ ibb.cu
#include <algorithm>
#include <cmath>       // @pinit.cu @links.cu
#include <cstdlib>
#include <cub/cub.cuh> // @pinit.cu

namespace {

// compute_link_q_C1(ip, x0, y0, z0, rad1_sq, q)
__device__ inline bool compute_link_q_C1(int ip, double x0, double y0, double z0,
                                      double radius_sq, double &q) {
    const double cx_dir = static_cast<double>(d_cix[ip]);
    const double cy_dir = static_cast<double>(d_ciy[ip]);
    // const double cz_dir = static_cast<double>(d_ciz[ip]);
    const double dx = x0 - d_pipe_xcenter;
    const double dy = y0 - d_pipe_ycenter;
    // const double dz = z0 - pz;
    const double a = cx_dir * cx_dir + cy_dir * cy_dir + cz_dir * cz_dir;
    const double b = 2.0 * (cx_dir * dx + cy_dir * dy + cz_dir * dz);
    const double c = dx * dx + dy * dy + dz * dz - radius_sq;
    const double disc = b * b - 4.0 * a * c;
    if (disc <= 0.0) return false;
    q = (-b - sqrt(disc)) / (2.0 * a);
    return (q >= 0.0 && q <= 1.0);
}

void allocate_pipe_arrays() {
    const size_t np = static_cast<size_t>(NPIPE);
    if (np == 0) return;

    h_pipe_force_x      = malloc_host_array<double>(np);
    h_pipe_force_y      = malloc_host_array<double>(np);
    h_pipe_force_z      = malloc_host_array<double>(np);
    h_pipe_torque_x     = malloc_host_array<double>(np);
    h_pipe_torque_y     = malloc_host_array<double>(np);
    h_pipe_torque_z     = malloc_host_array<double>(np);
    h_pipe_temp         = malloc_host_array<double>(np);
    h_pipe_heat         = malloc_host_array<double>(np);

    CHECK_CUDA_ERROR(cudaMalloc(&d_pipe_force_x,      np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pipe_force_y,      np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pipe_force_z,      np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pipe_torque_x,     np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pipe_torque_y,     np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pipe_torque_z,     np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pipe_temp,         np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pipe_heat,         np * sizeof(double)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_pipe_force_accum_x, np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pipe_force_accum_y, np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pipe_force_accum_z, np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pipe_torque_accum_x,  np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pipe_torque_accum_y,  np * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pipe_torque_accum_z,  np * sizeof(double)));

    // Grid markers & ownership (ibnode: 0 fluid, 1 outer cylinder, 2 inner cylinder)
    h_pipe_bnode       = malloc_host_array<int>(LXYZ);   //0 res fluid, 1 res particle
    h_pipe_bnode_owner = malloc_host_array<int>(LXYZ);   // current particle owner (-1 for fluid)
    CHECK_CUDA_ERROR(cudaMalloc(&d_pipe_bnode,       LXYZ * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pipe_bnode_owner, LXYZ * sizeof(int)));
}

void build_initial_pipe_bnode() {
    for (int idx = 0; idx < LXYZ; ++idx) {
        h_ibnode[idx] = 0;  //0 is fuild; 1 is the outer cylinder, 2 is the inner one
        h_ibnode_owner[idx] = -1; //pipe ID
    }

    // OUTER CYLINDER
    const double rad1_sq = PIPE_RAD1*PIPE_RAD1;
    for (int iz = 0; iz < LZ; ++iz) {
        for (int iy = 0; iy < LY; ++iy) {
            const double yy0 = static_cast<double>(iy) + 0.5 - pipe_ycenter;

            for (int ix = 0; ix < LX; ++ix) {
                const double xx0 = static_cast<double>(ix) + 0.5 - pipe_xcenter;
                const double rr0 = xx0 * xx0 + yy0 * yy0;

                int idx = iz * LXY + iy * LX + ix;
                if (rr0 >= rad1_sq) {
                    h_pipe_bnode[idx] = 1;
                    h_pipe_bnode_owner[idx] = 1; //assgin for outer cylinder
                    // break;
                }
            }
        }
    }

    // INNER CYLINDER
    const double rad2_sq = PIPE_RAD2*PIPE_RAD2;
    for (int iz = 0; iz < LZ; ++iz) {
        for (int iy = 0; iy < LY; ++iy) {
            const double yy0 = static_cast<double>(iy) + 0.5 - pipe_ycenter;

            for (int ix = 0; ix < LX; ++ix) {
                const double xx0 = static_cast<double>(ix) + 0.5 - pipe_xcenter;
                const double rr0 = xx0 * xx0 + yy0 * yy0;

                int idx = iz * LXY + iy * LX + ix;
                if (rr0 <= rad2_sq) {
                    h_pipe_bnode[idx] = 2;
                    h_pipe_bnode_owner[idx] = 2; //assgin for inner cylinder
                    // break;
                }
            }
        }
    }

    CHECK_CUDA_ERROR(cudaMemcpy(d_pipe_bnode,       h_pipe_bnode,       LXYZ * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pipe_bnode_owner, h_pipe_bnode_owner, LXYZ * sizeof(int), cudaMemcpyHostToDevice));
}

// pipe_link_count_kernel<<<blocks, threads>>>(d_pipe_link_count,
//                                        d_pipe_bnode, d_pipe_bnode_owner);
__global__ void pipe_link_count_kernel(int *count,
                                  const int *ibnode,
                                  const int *owner) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= LXYZ) return;
    if (ibnode[idx] == 1 || ibnode[idx] == 2) {
        count[idx] = 0;
        return;
    }

    // DETERMINE the global position
    const int iz = idx / LXY;
    const int rem = idx - iz * LXY;
    const int iy = rem / LX;
    const int ix = rem - iy * LX;

    const double x0 = static_cast<double>(ix) + 0.5;
    const double y0 = static_cast<double>(iy) + 0.5;
    const double z0 = static_cast<double>(iz) + 0.5;
    const double rad1_sq = PIPE_RAD1 * PIPE_RAD1;
    const double rad2_sq = PIPE_RAD2 * PIPE_RAD2;

    int cnt = 0;
    for (int ip = 1; ip < NPOP; ++ip) {
        const int imove = ix + d_cix[ip];
        const int jmove = iy + d_ciy[ip];
        // const int nz = iz + d_ciz[ip];
        if (imove < 0 || imove >= LX || jmove < 0 || jmove >= LY ) continue;
        const int nidx = nz * LXY + jmove * LX + imove;
        const int pid = owner[nidx];///
        if (pid < 0) continue;

        double q = 0.0;
        if (pid == 1) {
            if (compute_link_q_C1(ip, x0, y0, z0, rad1_sq, q)) {
                cnt++;
            }
        }
        else if (pid == 2)
        {
            if (compute_link_q_C2(ip, x0, y0, z0, rad2_sq, q)) {
                cnt++;
            }
        }       
    }
    count[idx] = cnt;
}

// pipe_link_fill_kernel<<<blocks, threads>>>(d_pipe_link_count, d_pipe_link_offset,
//                                       d_pipe_bnode, d_pipe_bnode_owner,
//                                       d_pipe_links);
__global__ void pipe_link_fill_kernel(const int *count, const int *offset,
                                 const int *ibnode,
                                 const int *owner
                                 PipeLink *links) {
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

void init_pipes() {
    init_pipe_paras();

    if (!ACTIVATE_PARTICLES || NPIPE == 0) {
        // Ensure ibnode arrays exist even when ACTIVATE_PARTICLES is 0
        h_pipe_bnode = malloc_host_array<int>(LXYZ);
        h_pipe_bnode_owner = malloc_host_array<int>(LXYZ);
        if (!h_pipe_bnode || !h_pipe_bnode_owner) {
            fprintf(stderr, "Failed to allocate ibnode buffers.\n");
            exit(EXIT_FAILURE);
        }
        std::fill(h_pipe_bnode, h_pipe_bnode + LXYZ, 0);  // void fill(start_loc, end_loc, value)
        std::fill(h_pipe_bnode_owner, h_pipe_bnode_owner + LXYZ, -1);

        CHECK_CUDA_ERROR(cudaMalloc(&d_pipe_bnode,       LXYZ * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_pipe_bnode_owner, LXYZ * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMemcpy(d_pipe_bnode,       h_pipe_bnode,       LXYZ * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_pipe_bnode_owner, h_pipe_bnode_owner, LXYZ * sizeof(int), cudaMemcpyHostToDevice));

        return;
    }

    allocate_pipe_arrays();

    std::fill(h_pipe_theta_x,      h_pipe_theta_x      + NPIPE, 0.0);
    std::fill(h_pipe_theta_y,      h_pipe_theta_y      + NPIPE, 0.0);
    std::fill(h_pipe_theta_z,      h_pipe_theta_z      + NPIPE, 0.0);
    std::fill(h_pipe_force_x,      h_pipe_force_x      + NPIPE, 0.0);
    std::fill(h_pipe_force_y,      h_pipe_force_y      + NPIPE, 0.0);
    std::fill(h_pipe_force_z,      h_pipe_force_z      + NPIPE, 0.0);
    std::fill(h_pipe_torque_x,     h_pipe_torque_x     + NPIPE, 0.0);
    std::fill(h_pipe_torque_y,     h_pipe_torque_y     + NPIPE, 0.0);
    std::fill(h_pipe_torque_z,     h_pipe_torque_z     + NPIPE, 0.0);
    std::fill(h_pipe_heat,         h_pipe_heat         + NPIPE, 0.0);

    const double initial_temp = 0.0;
    std::fill(h_pipe_temp, h_pipe_temp + NIPE, initial_temp);
    std::fill_n(h_pipe_force_accum_x   = malloc_host_array<double>(NPIPE), NPIPE, 0.0);  //void fill_n(start_loc, N , value)
    std::fill_n(h_pipe_force_accum_y   = malloc_host_array<double>(NPIPE), NPIPE, 0.0);
    std::fill_n(h_pipe_force_accum_z   = malloc_host_array<double>(NPIPE), NPIPE, 0.0);
    std::fill_n(h_pipe_torque_accum_x  = malloc_host_array<double>(NPIPE), NPIPE, 0.0);
    std::fill_n(h_pipe_torque_accum_y  = malloc_host_array<double>(NPIPE), NPIPE, 0.0);
    std::fill_n(h_pipe_torque_accum_z  = malloc_host_array<double>(NPIPE), NPIPE, 0.0);

    CHECK_CUDA_ERROR(cudaMemcpy(d_pipe_theta_x,      h_pipe_theta_x,      NPIPE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pipe_theta_y,      h_pipe_theta_y,      NPIPE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pipe_theta_z,      h_pipe_theta_z,      NPIPE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pipe_force_x,      h_pipe_force_x,      NPIPE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pipe_force_y,      h_pipe_force_y,      NPIPE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pipe_force_z,      h_pipe_force_z,      NPIPE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pipe_torque_x,     h_pipe_torque_x,     NPIPE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pipe_torque_y,     h_pipe_torque_y,     NPIPE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pipe_torque_z,     h_pipe_torque_z,     NPIPE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pipe_temp,         h_pipe_temp,         NPIPE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pipe_heat,         h_pipe_heat,         NPIPE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pipe_article_force_accum_x, h_pipe_article_force_accum_x, NPIPE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pipe_article_force_accum_y, h_pipe_article_force_accum_y, NPIPE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pipe_article_force_accum_z, h_pipe_article_force_accum_z, NPIPE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pipe_article_torque_accum_x,  h_pipe_article_torque_accum_x,  NPIPE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pipe_article_torque_accum_y,  h_pipe_article_torque_accum_y,  NPIPE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pipe_article_torque_accum_z,  h_pipe_article_torque_accum_z,  NPIPE * sizeof(double), cudaMemcpyHostToDevice));

    build_initial_pipe_bnode();
}

void build_pipe_links() {
    if (!ACTIVATE_PIPE || NPIPE == 0) return;

    // Clear current ibnode and owner.
    CHECK_CUDA_ERROR(cudaMemset(d_pipe_bnode, 0, LXYZ * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemset(d_pipe_bnode_owner, 0xFF, LXYZ * sizeof(int))); // 0xFF is -1

    const double rad1_sq = PIPE_RAD1 * PIPE_RAD1;
    const double rad2_sq = PIPE_RAD2 * PIPE_RAD2;

    const dim3 block_ib(8, 8, 8);


    // check wheter "d_pipe_link_count" and "d_pipe_link_offset" is nullptr
    if (!d_pipe_link_count) {
        CHECK_CUDA_ERROR(cudaMalloc(&d_pipe_link_count, LXYZ * sizeof(int)));
    }
    if (!d_pipe_link_offset) {
        CHECK_CUDA_ERROR(cudaMalloc(&d_pipe_link_offset, LXYZ * sizeof(int)));
    }
    size_t scan_bytes = 0;
    CHECK_CUDA_ERROR(cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes,
                                                   d_pipe_link_count, d_pipe_link_offset, LXYZ));
    if (scan_bytes > d_pipe_link_scan_bytes) {
        if (d_pipe_link_scan_tmp) {
            CHECK_CUDA_ERROR(cudaFree(d_pipe_link_scan_tmp));
        }
        CHECK_CUDA_ERROR(cudaMalloc(&d_pipe_link_scan_tmp, scan_bytes));
        d_pipe_link_scan_bytes = scan_bytes;
    }

    const int threads = 256;
    const int blocks = (LXYZ + threads - 1) / threads;

    int last_offset = 0;
    int last_count = 0;
    // Link counting.
    pipe_link_count_kernel<<<blocks, threads>>>(d_pipe_link_count,
                                           d_pipe_bnode, d_pipe_bnode_owner);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cub::DeviceScan::ExclusiveSum(d_pipe_link_scan_tmp, d_pipe_link_scan_bytes,
                                                   d_pipe_link_count, d_pipe_link_offset, LXYZ));


    CHECK_CUDA_ERROR(cudaMemcpy(&last_offset, d_pipe_link_offset + (LXYZ - 1), // last element in "d_pipe_link_offset"?
                                sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(&last_count, d_pipe_link_count + (LXYZ - 1),
                                sizeof(int), cudaMemcpyDeviceToHost));
    num_pipe_links = last_offset + last_count;

    if (num_pipe_links > pipe_link_capacity) {
        if (d_pipe_links) {
            CHECK_CUDA_ERROR(cudaFree(d_pipe_links));
        }
        CHECK_CUDA_ERROR(cudaMalloc(&d_pipe_links,
                                    num_pipe_links * sizeof(PipeLink)));
        pipe_link_capacity = num_pipe_links;
    }
    if (num_pipe_links > 0) {
        pipe_link_fill_kernel<<<blocks, threads>>>(d_pipe_link_count, d_pipe_link_offset,
                                              d_pipe_bnode, d_pipe_bnode_owner,
                                              d_pipe_links);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}