#include "lbm.h"

__global__ void collision_BGK(double *f, double *force_realx, double *force_realy,
                             double *force_realz, double *rho, double *ux,
                             double *uy, double *uz) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= LX || iy >= LY || iz >= LZ) return;

    int idx = (iz * LXY) + (iy * LX) + ix;
    double fx = force_realx[idx];
    double rho_local = rho[idx];
    double ux_local = ux[idx];
    double uy_local = uy[idx];
    double uz_local = uz[idx];
    const double RT = 1.0 / 3.0;
    const double force_coeff = (1.0 - 0.5 / d_tau);
    const double u2 = ux_local * ux_local + uy_local * uy_local + uz_local * uz_local;
    const double uv = u2 / RT;

    for (int ip = 0; ip < NPOP; ip++) {
        double eu = (d_cix[ip] * ux_local + d_ciy[ip] * uy_local + d_ciz[ip] * uz_local) / RT;
        double feq = d_tp[ip] * (rho_local + eu + 0.5 * (eu * eu - uv));

        double fpop = f[ip * LXYZ + idx];
        double fchange = -(fpop - feq) / d_tau;
        double feq_rho0 = d_tp[ip] * (1.0 + eu + 0.5 * (eu * eu - uv));
        double cuF = (d_cix[ip] - ux_local) * fx;
        fchange += force_coeff * cuF / RT * feq_rho0;
        f[ip * LXYZ + idx] = fpop + fchange;
    }
}

__global__ void collision_BGK_scalar(double *g, double *force_realx, double *force_realy,
                                    double *force_realz, double *rho, double *ux,
                                    double *uy, double *uz, double *phi) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= LX || iy >= LY || iz >= LZ) return;

    int idx = (iz * LXY) + (iy * LX) + ix;
    double fx = force_realx[idx];
    double fy = force_realy[idx];
    double fz = force_realz[idx];
    double rho_local = rho[idx];
    double phi_local = phi[idx];
    double ux_local = ux[idx];
    double uy_local = uy[idx];
    double uz_local = uz[idx];
    const double RT = 1.0 / 3.0;
    const double u2 = ux_local * ux_local + uy_local * uy_local + uz_local * uz_local;
    const double uv = u2 / RT;
    const double lambda0 = 1.0 - d_tp[0];

    double gneqx = 0.0;
    double gneqy = 0.0;
    double gneqz = 0.0;
    for (int ip = 0; ip < NPOP; ip++) {
        double eu = (d_cix[ip] * ux_local + d_ciy[ip] * uy_local + d_ciz[ip] * uz_local) / RT;
        double geq = d_tp[ip] * phi_local * (1.0 + eu + 0.5 * (eu * eu - uv));
        double geqnew = (ip == 0)
                            ? (geq - lambda0 * phi_local * rho_local)
                            : (geq + d_tp[ip] * phi_local * rho_local);
        double diff = g[ip * LXYZ + idx] - geqnew;
        gneqx += d_cix[ip] * diff;
        gneqy += d_ciy[ip] * diff;
        gneqz += d_ciz[ip] * diff;
    }

    const double Ralph_coeff = (1.0 - 1.0 / (2.0 * d_tauc));
    const double grad_denom = rho_local / 3.0 + 2.0 * d_tauc / 3.0;
    for (int ip = 0; ip < NPOP; ip++) {
        double eb = d_cix[ip] * fx + d_ciy[ip] * fy + d_ciz[ip] * fz;
        double gneq = d_cix[ip] * gneqx + d_ciy[ip] * gneqy + d_ciz[ip] * gneqz;
        double gradt = -(2.0 * gneq + eb * phi_local) / grad_denom;
        double Ralph = d_tp[ip] * Ralph_coeff * (gradt * rho_local + 3.0 * phi_local * eb);
        double eu = (d_cix[ip] * ux_local + d_ciy[ip] * uy_local + d_ciz[ip] * uz_local) / RT;
        double geq = d_tp[ip] * phi_local * (1.0 + eu + 0.5 * (eu * eu - uv));
        double geqnew = (ip == 0)
                            ? (geq - lambda0 * phi_local * rho_local)
                            : (geq + d_tp[ip] * phi_local * rho_local);
        double gpop = g[ip * LXYZ + idx];
        double gchange = -(gpop - geqnew) / d_tauc + Ralph;
        g[ip * LXYZ + idx] = gpop + gchange;
    }

}

__global__ void compute_Ked(double *f, double *force_realx, double *force_realy,
                            double *force_realz, double *rho, double *ux,
                            double *uy, double *uz, double *Ked) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= LX || iy >= LY || iz >= LZ) return;

    int idx = (iz * LXY) + (iy * LX) + ix;
    double fx = force_realx[idx];
    double fy = force_realy[idx];
    double fz = force_realz[idx];
    double rho_local = rho[idx];
    double ux_local = ux[idx];
    double uy_local = uy[idx];
    double uz_local = uz[idx];
    const double RT = 1.0 / 3.0;
    const double u2 = ux_local * ux_local + uy_local * uy_local + uz_local * uz_local;
    const double uv = u2 / RT;
    const double bu = fx * ux_local + fy * uy_local + fz * uz_local;

    double tau_xx = 0.0;
    double tau_xy = 0.0;
    double tau_xz = 0.0;
    double tau_yy = 0.0;
    double tau_yz = 0.0;
    double tau_zz = 0.0;

    for (int ip = 0; ip < NPOP; ip++) {
        double eu = (d_cix[ip] * ux_local + d_ciy[ip] * uy_local + d_ciz[ip] * uz_local) / RT;
        double feq = d_tp[ip] * (rho_local + eu + 0.5 * (eu * eu - uv));
        double fneq = f[ip * LXYZ + idx] - feq;

        double cidotF = d_cix[ip] * fx + d_ciy[ip] * fy + d_ciz[ip] * fz;
        double feq_term = 0.5 * (cidotF - bu) * feq / RT;

        tau_xx += d_cix[ip] * d_cix[ip] * (fneq + feq_term);
        tau_xy += d_cix[ip] * d_ciy[ip] * (fneq + feq_term);
        tau_xz += d_cix[ip] * d_ciz[ip] * (fneq + feq_term);
        tau_yy += d_ciy[ip] * d_ciy[ip] * (fneq + feq_term);
        tau_yz += d_ciy[ip] * d_ciz[ip] * (fneq + feq_term);
        tau_zz += d_ciz[ip] * d_ciz[ip] * (fneq + feq_term);
    }

    double coeff = -(1.0 - 0.5 / d_tau);
    tau_xx *= coeff;
    tau_xy *= coeff;
    tau_xz *= coeff;
    tau_yy *= coeff;
    tau_yz *= coeff;
    tau_zz *= coeff;

    double S_xx = tau_xx / (2.0 * d_visc);
    double S_xy = tau_xy / (2.0 * d_visc);
    double S_xz = tau_xz / (2.0 * d_visc);
    double S_yy = tau_yy / (2.0 * d_visc);
    double S_yz = tau_yz / (2.0 * d_visc);
    double S_zz = tau_zz / (2.0 * d_visc);

    Ked[idx] = 0.5 * d_visc * (4.0 * (S_xx * S_xx + S_yy * S_yy + S_zz * S_zz) +
                               8.0 * (S_xy * S_xy + S_xz * S_xz + S_yz * S_yz));
}

__global__ void compute_ted(double *g, double *force_realx, double *force_realy,
                            double *force_realz, double *rho, double *ux,
                            double *uy, double *uz, double *phi, double *ted) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= LX || iy >= LY || iz >= LZ) return;

    int idx = (iz * LXY) + (iy * LX) + ix;
    double fx = force_realx[idx];
    double fy = force_realy[idx];
    double fz = force_realz[idx];
    double rho_local = rho[idx];
    double phi_local = phi[idx];
    double ux_local = ux[idx];
    double uy_local = uy[idx];
    double uz_local = uz[idx];
    const double RT = 1.0 / 3.0;
    const double u2 = ux_local * ux_local + uy_local * uy_local + uz_local * uz_local;
    const double uv = u2 / RT;
    const double lambda0 = 1.0 - d_tp[0];

    double gneqx = 0.0;
    double gneqy = 0.0;
    double gneqz = 0.0;
    for (int ip = 0; ip < NPOP; ip++) {
        double eu = (d_cix[ip] * ux_local + d_ciy[ip] * uy_local + d_ciz[ip] * uz_local) / RT;
        double geq = d_tp[ip] * phi_local * (1.0 + eu + 0.5 * (eu * eu - uv));
        double geqnew = (ip == 0)
                            ? (geq - lambda0 * phi_local * rho_local)
                            : (geq + d_tp[ip] * phi_local * rho_local);
        double diff = g[ip * LXYZ + idx] - geqnew;
        gneqx += d_cix[ip] * diff;
        gneqy += d_ciy[ip] * diff;
        gneqz += d_ciz[ip] * diff;
    }

    double p = rho_local * RT;
    double denominator = 2.0 * d_tauc * RT + p;
    double tx = -(2.0 * gneqx + phi_local * fx) / denominator;
    double ty = -(2.0 * gneqy + phi_local * fy) / denominator;
    double tz = -(2.0 * gneqz + phi_local * fz) / denominator;

    ted[idx] = (d_diff > 0.0) ? d_diff * (tx * tx + ty * ty + tz * tz) : 0.0;
}
