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
        //     int pid;  //Cylinder ID of link
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

    // x_b, as f_collide is the post collision distributions,
    // they can be used directly without thinking of the locations after propagation‌
    const int ix = link.cell_i;
    const int iy = link.cell_j;
    const int iz = link.cell_k;
    // x_bb
    const int ix1 = ix - d_cix[ip];
    const int iy1 = iy - d_ciy[ip];
    // const int iz1 = iz - d_ciz[ip];
    const int iz1 = ( iz - d_ciz[ip] + LZ ) % LZ; // z-periodic
    // x_bbb
    const int ix2 = ix - 2 * d_cix[ip];
    const int iy2 = iy - 2 * d_ciy[ip];
    // const int iz2 = iz - 2.0 * d_ciz[ip]
    const int iz2 = ( iz - 2 * d_ciz[ip] + LZ ) % LZ; // z-periodic

    const double cx = static_cast<double>(ix) + 0.5 + q * d_cix[ip];  // surface cordi
    const double cy = static_cast<double>(iy) + 0.5 + q * d_ciy[ip];  // surface cordi
    const double cz = static_cast<double>(iz) + 0.5 + q * d_ciz[ip];  // surface cordi
    const double px = d_pipe_xcenter;
    const double py = d_pipe_ycenter;
    // const double pz = ppos_z[pid];
    const double relx = cx - px;
    const double rely = cy - py;
    // const double relz = cz - pz;

    double uwx = 0.0; // d_pipe_omega1
    double uwy = 0.0;
    double uwz = 0.0;
    if (pid == 1) {
        uwx = - rely * d_pipe_omega1; // d_pipe_omega1
        uwy =   relx * d_pipe_omega1;
    } else if (pid ==2){
        uwx = - rely * d_pipe_omega2; // d_pipe_omega2
        uwy =   relx * d_pipe_omega2;
    }
    // const double uwz = pvel_z[pid] + (rely * pomega_x[pid] - relx * pomega_y[pid]);
    const double evel = d_cix[ip] * uwx + d_ciy[ip] * uwy + d_ciz[ip] * uwz; // Velocity projection e_{alpha} \cdot u
    const double RT = 1.0 / 3.0;
    const double momentum_term = 2.0 * d_tp[ip] * RHO0 * evel / RT;

    const int idx_bb  = iz1 * LXY + iy1 * LX + ix1; // x_bb
    const int idx_bbb = iz2 * LXY + iy2 * LX + ix2; // x_bbb
    // q >0.5 \bar{a} is ip, a is the opposite direction, in my thesis
    const double f_b       = f_collide[ ip * LXYZ + idx    ]; // f_{\bar{a}}(x_b)
    const double f_opp_bb  = f_collide[opp * LXYZ + idx_bb ]; // f_{a}(x_bb)
    const double f_opp_bbb = f_collide[opp * LXYZ + idx_bbb]; // f_{a}(x_bbb)

    const double f_bb  = f_collide[ ip * LXYZ + idx_bb  ]; // f_{\bar{a}}(x_bb)
    const double f_bbb = f_collide[ ip * LXYZ + idx_bbb ]; // f_{\bar{a}}(x_bbb)

    bool handled = false;
    // Linear interpolation
    // if (ix1 >= 0 && ix1 < LX && iy1 >= 0 && iy1 < LY && iz1 >= 0 && iz1 < LZ &&
    //     ix2 >= 0 && ix2 < LX && iy2 >= 0 && iy2 < LY && iz2 >= 0 && iz2 < LZ) {
    if (ix1 >= 0 && ix1 < LX && iy1 >= 0 && iy1 < LY &&
        ix2 >= 0 && ix2 < LX && iy2 >= 0 && iy2 < LY ) {
        // const int idx1 = iz1 * LXY + iy1 * LX + ix1;
        if (ibnode[idx_bb] == 0 && ibnode[idx_bb]) {
            // const double f_here = f_collide[ip * LXYZ + idx]; //f_b
            if (q <= 0.5) {
                const double c1 = q * (1.0 + 2.0*q);
                const double c2 = 1.0 - q * q;
                const double c3 = 1.0 -c1 - c2;
                f[opp * LXYZ + idx] = c1 * f_b + c2 * f_bb + c3 * f_bbb - momentum_term;
            } else {
                const double c1 = 1.0 / q / (1.0 +2.0 * q);
                const double c2 = (2.0 * q - 1.0) / q;
                const double c3 = 1.0 -c1 - c2;
                f[opp * LXYZ + idx] = c1 * (f_b - momentum_term) + c2 * f_opp_bb + c3 * f_opp_bbb;
            }
            handled = true;
        }
    }
    // // Simple bounce-back
    // if (!handled && ibnode[idx] == 0) {
    //     const double f_here = f_collide[ip * LXYZ + idx];
    //     f[opp * LXYZ + idx] = f_here - momentum_term; 
    //     handled = true;
    // }

    // if (!handled) {
    //     // Fall back to simple reflection without slip correction.
    //     const double f_here = f_collide[ip * LXYZ + idx];
    //     f[opp * LXYZ + idx] = f_here;
    //     printf("bounce_back_particles: unprocessed link tid=%d, pid=%d, ix=%d, iy=%d\n", 
    //         tid, pid, ix, iy);
    // }
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
        //     int pid;  //Cylinder ID of link
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

    // x_bb
    const int ix1 = ix - d_cix[ip];
    const int iy1 = iy - d_ciy[ip];
    const int iz1 = iz - d_ciz[ip];
    const int idx_bb = iz1 * LXY + iy1 * LX + ix1;
    // x_bbb
    const int ix2 = ix - 2 * d_cix[ip];
    const int iy2 = iy - 2 * d_ciy[ip];
    const int iz2 = iz - 2 * d_ciz[ip];
    const int idx_bbb = iz2 * LXY + iy2 * LX + ix2;

    const double density = 1.0; //rho[idx];
    const double Tp = 0.0; //ptemp[pid];
    const double amp = 2.0 * d_tp[ip] * Tp * (1.0 + density);

    double value = 0.0;
    // if (ix1 >= 0 && ix1 < LX && iy1 >= 0 && iy1 < LY &&
    //     iz1 >= 0 && iz1 < LZ ) {
    if (ix1 >= 0 && ix1 < LX && iy1 >= 0 && iy1 < LY &&
        ix2 >= 0 && ix2 < LX && iy2 >= 0 && iy2 < LY ) {
        if (q <= 0.5) {
            const double g_here = g_collide[ip * LXYZ + idx];
            const double g_upstream = g_collide[ip * LXYZ + idx_bb];
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
    if (!ACTIVATE_PIPE || NPIPE == 0 || num_pipe_links == 0) return;
    const int threads = 128;
    const int blocks = (num_pipe_links + threads - 1) / threads;
    pipe_bb_kernel<<<blocks, threads>>>(d_pipe_links, num_pipe_links,
                                                     d_f,
                                                     d_f_collide,
                                                     d_pipe_bnode);
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
