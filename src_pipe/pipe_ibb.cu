#include "pipe.h"

namespace {


        // bounce_back_particles_kernel<<<blocks, threads>>>(d_particle_links, num_particle_links,
        //                                              d_f,
        //                                              d_f_collide,
        //                                              d_ibnode);
        // struct PipeLink {
        //     int cell_i; //fuild point position x
        //     int cell_j;
        //     int cell_k;
        //     int dir;  //direction ip
        //     int pid;  //particle ID of link
        //     double q; //intersection of link and particle boundary (0-1)
        //     double rx; //q * cix[ip]
        //     double ry;
        //     double rz;
        // };
        // SHOULD consider the owner
__global__ void pipe_bb_kernel(const PipeLink *links, int num_links,
                                             double *f,
                                             const double *f_collide,
                                             const int *ibnode) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_links) return;

    const PipeLink link = links[tid];
    const int ip = link.dir;
    const int opp = d_opp[ip];
    const int idx = link.cell_k * LXY + link.cell_j * LX + link.cell_i;
    const int pid = link.pid;
    const double q = link.q;

    const int ix = link.cell_i;
    const int iy = link.cell_j;
    const int iz = link.cell_k;
    const int ix1 = ix - d_cix[ip];
    const int iy1 = iy - d_ciy[ip];
    const int iz1 = iz - d_ciz[ip];

    const double cx = static_cast<double>(ix) + 0.5 + q * d_cix[ip];
    const double cy = static_cast<double>(iy) + 0.5 + q * d_ciy[ip];
    const double cz = static_cast<double>(iz) + 0.5 + q * d_ciz[ip];
    const double px = ppos_x[pid];
    const double py = ppos_y[pid];
    const double pz = ppos_z[pid];
    const double relx = cx - px;
    const double rely = cy - py;
    const double relz = cz - pz;

    const double uwx = pvel_x[pid] + (relz * pomega_y[pid] - rely * pomega_z[pid]);
    const double uwy = pvel_y[pid] + (relx * pomega_z[pid] - relz * pomega_x[pid]);
    const double uwz = pvel_z[pid] + (rely * pomega_x[pid] - relx * pomega_y[pid]);
    const double evel = d_cix[ip] * uwx + d_ciy[ip] * uwy + d_ciz[ip] * uwz; // Velocity projection e_{alpha} \cdot u
    const double RT = 1.0 / 3.0;
    const double momentum_term = 2.0 * d_tp[ip] * RHO0 * evel / RT;

    bool handled = false;
    // Linear interpolation
    if (ix1 >= 0 && ix1 < LX && iy1 >= 0 && iy1 < LY && iz1 >= 0 && iz1 < LZ) {
        const int idx1 = iz1 * LXY + iy1 * LX + ix1;
        if (ibnode[idx1] == 0) {
            const double f_here = f_collide[ip * LXYZ + idx];
            if (q < 0.5) {
                const double f_upstream = f_collide[ip * LXYZ + idx1];
                f[opp * LXYZ + idx] = 2.0 * q * f_here + (1.0 - 2.0 * q) * f_upstream - momentum_term;
            } else {
                const double f_opposite_collide = f_collide[opp * LXYZ + idx];
                const double inv_2q = 0.5 / q;
                f[opp * LXYZ + idx] = (f_here - momentum_term) * inv_2q +
                                     (2.0 * q - 1.0) * inv_2q * f_opposite_collide;
            }
            handled = true;
        }
    }
    // Simple bounce-back
    if (!handled && ibnode[idx] == 0) {
        const double f_here = f_collide[ip * LXYZ + idx];
        f[opp * LXYZ + idx] = f_here - momentum_term; 
        handled = true;
    }

    if (!handled) {
        // Fall back to simple reflection without slip correction.
        const double f_here = f_collide[ip * LXYZ + idx];
        f[opp * LXYZ + idx] = f_here;
        printf("bounce_back_particles: unprocessed link tid=%d, pid=%d, ix=%d, iy=%d\n", 
            tid, pid, ix, iy);
    }
}


    // pipe_bb_thermal_kernel<<<blocks, threads>>>(d_pipe_links, num_pipe_links,
    //                                                d_g,
    //                                                d_g_collide,
    //                                                d_pipe_bnode);
        // struct PipeLink {
        //     int cell_i; //fuild point position x
        //     int cell_j;
        //     int cell_k;
        //     int dir;  //direction ip
        //     int pid;  //particle ID of link
        //     double q; //intersection of link and particle boundary (0-1)
        //     double rx; //q * cix[ip]
        //     double ry;
        //     double rz;
        // };
        // SHOULD consider the owner
__global__ void pipe_bb_thermal_kernel(const PipeLink *links, int num_links,
                                           double *g,
                                           const double *g_collide,
                                           const int *ibnode_curr) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_links) return;

    const PipeLink link = links[tid];
    const int ip = link.dir;
    const int opp = d_opp[ip];
    const int ix = link.cell_i;
    const int iy = link.cell_j;
    const int iz = link.cell_k;
    const int idx = iz * LXY + iy * LX + ix;
    const int pid = link.pid;
    const double q = link.q;

    const int ix1 = ix - d_cix[ip];
    const int iy1 = iy - d_ciy[ip];
    const int iz1 = iz - d_ciz[ip];
    const int idx1 = iz1 * LXY + iy1 * LX + ix1;

    const double density = rho[idx];
    const double Tp = ptemp[pid];
    const double amp = 2.0 * d_tp[ip] * Tp * (1.0 + density);

    double value = 0.0;
    if (ix1 >= 0 && ix1 < LX && iy1 >= 0 && iy1 < LY &&
        iz1 >= 0 && iz1 < LZ && ibnode_prev[idx1] == 0) {
        if (q <= 0.5) {
            const double g_here = g_collide[ip * LXYZ + idx];
            const double g_upstream = g_collide[ip * LXYZ + idx1];
            value = -(2.0 * q * g_here + (1.0 - 2.0 * q) * g_upstream) + amp;
        } else {
            const double g_here = g_collide[ip * LXYZ + idx];
            const double g_opp_here = g_collide[opp * LXYZ + idx];
            const double inv_2q = 0.5 / q;
            value = -g_here * inv_2q + amp * inv_2q + (2.0 * q - 1.0) * inv_2q * g_opp_here;
        }
    } else if (ibnode_curr[idx] == 0) {
        const double g_here = g_collide[ip * LXYZ + idx];
        value = -g_here + amp;
    } else {
        printf("bounce_back_thermal: error link tid=%d, pid=%d, ix=%d, iy=%d, q=%f\n", 
            tid, pid, ix, iy, q);
        return;
    }

    g[opp * LXYZ + idx] = value;
}

} // namespace

void pipe_bounce_back() {
    if (!ACTIVATE_PARTICLES || NPIPE == 0 || num_pipe_links == 0) return;
    const int threads = 128;
    const int blocks = (num_particle_links + threads - 1) / threads;
    pipe_bb_kernel<<<blocks, threads>>>(d_pipe_links, num_pipe_links,
                                                     d_f,
                                                     d_f_collide,
                                                     d_ibnode);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

void pipe_bounce_back_thermal() {
    if (!ACTIVATE_PARTICLES || NPIPE == 0 || num_pipe_links == 0) return;
    const int threads = 128;
    const int blocks = (num_pipe_links + threads - 1) / threads;
    pipe_bb_thermal_kernel<<<blocks, threads>>>(d_pipe_links, num_pipe_links,
                                                   d_g,
                                                   d_g_collide,
                                                   d_pipe_bnode);
    CHECK_CUDA_ERROR(cudaGetLastError());
}
