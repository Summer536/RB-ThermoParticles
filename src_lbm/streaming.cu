#include "lbm.h"

__global__ void streaming(double *f, double *f_temp) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= LX || iy >= LY || iz >= LZ) return;

    int idx = (iz * LXY) + (iy * LX) + ix;

    for (int ip = 0; ip < NPOP; ++ip) {
        int sx = ix - d_cix[ip];
        int sy = iy - d_ciy[ip];
        int sz = iz - d_ciz[ip];
        if (sx >= 0 && sx < LX && sy >= 0 && sy < LY && sz >= 0 && sz < LZ) {
            int src = (sz * LXY) + (sy * LX) + sx;
            f_temp[ip * LXYZ + idx] = f[ip * LXYZ + src];
        } else {
            int opp = d_opp[ip];
            f_temp[ip * LXYZ + idx] = f[opp * LXYZ + idx];
        }
    }
}

__global__ void streaming_scalar(double *g, double *g_temp, double *rho) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= LX || iy >= LY || iz >= LZ) return;

    int idx = (iz * LXY) + (iy * LX) + ix;
    double density = rho[idx];

    for (int ip = 0; ip < NPOP; ++ip) {
        int sx = ix - d_cix[ip];
        int sy = iy - d_ciy[ip];
        int sz = iz - d_ciz[ip];
        if (sx >= 0 && sx < LX && sy >= 0 && sy < LY && sz >= 0 && sz < LZ) {
            int src = (sz * LXY) + (sy * LX) + sx;
            g_temp[ip * LXYZ + idx] = g[ip * LXYZ + src];
            continue;
        }

        int opp = d_opp[ip];
        if (sz < 0 && d_ciz[ip] > 0) {
            g_temp[ip * LXYZ + idx] = g[opp * LXYZ + idx]; // z=0: adiabatic wall
        } else if (sz >= LZ && d_ciz[ip] < 0) {
            g_temp[ip * LXYZ + idx] = g[opp * LXYZ + idx]; // z=LZ: adiabatic wall
        } else if (sy < 0 && d_ciy[ip] > 0) {
            g_temp[ip * LXYZ + idx] = g[opp * LXYZ + idx]; // y=0: adiabatic wall
        } else if (sy >= LY && d_ciy[ip] < 0) {
            g_temp[ip * LXYZ + idx] = g[opp * LXYZ + idx]; // y=LY: adiabatic wall
        } else if (sx < 0 && d_cix[ip] > 0) {
            g_temp[ip * LXYZ + idx] = -g[opp * LXYZ + idx] +
                                      2.0 * d_tp[ip] * d_tHot * (1.0 + density); // x=0: hot plate
        } else if (sx >= LX && d_cix[ip] < 0) {
            g_temp[ip * LXYZ + idx] = -g[opp * LXYZ + idx] +
                                      2.0 * d_tp[ip] * d_tCold * (1.0 + density); // x=LX: cold plate
        } else {
            g_temp[ip * LXYZ + idx] = g[opp * LXYZ + idx];
        }
    }
}
