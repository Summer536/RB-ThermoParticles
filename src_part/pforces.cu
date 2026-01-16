#include "particle.h"
#include <cmath>

namespace {

__global__ void repulsive_pair_kernel(const double *ppos_x, const double *ppos_y,
                                      const double *ppos_z,
                                      double *force_x, double *force_y, double *force_z,
                                      int np, double radius, double coeff, double min_gap) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int q = blockIdx.y * blockDim.y + threadIdx.y;
    if (p >= np || q >= np || q <= p) return;

    double dx = ppos_x[p] - ppos_x[q];
    double dy = ppos_y[p] - ppos_y[q];
    double dz = ppos_z[p] - ppos_z[q];
    double dist_sq = dx * dx + dy * dy + dz * dz;
    if (dist_sq <= 1.0e-12) return;

    double dist = sqrt(dist_sq);
    double hgap = dist - 2.0 * radius;
    if (hgap >= min_gap) return;

    double overlap = (hgap < 0.0) ? -hgap : 0.0;
    double factor = (min_gap - hgap) / min_gap;
    double lubforce = coeff * (factor * factor / REP_STF0 +
                               overlap / (min_gap * REP_STF1));
    double nx = dx / dist;
    double ny = dy / dist;
    double nz = dz / dist;

    atomicAdd(&force_x[p], lubforce * nx);
    atomicAdd(&force_y[p], lubforce * ny);
    atomicAdd(&force_z[p], lubforce * nz);
    atomicAdd(&force_x[q], -lubforce * nx);
    atomicAdd(&force_y[q], -lubforce * ny);
    atomicAdd(&force_z[q], -lubforce * nz);
}

__global__ void repulsive_wall_kernel(const double *ppos_x, const double *ppos_y,
                                      const double *ppos_z,
                                      double *force_x, double *force_y, double *force_z,
                                      int np, double radius, double coeff, double min_gap_w) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= np) return;

    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;

    auto apply_wall_force = [&](double gap, double nx, double ny, double nz) {
        if (gap >= min_gap_w) return;
        double overlap = (gap < 0.0) ? -gap : 0.0;
        double factor = (min_gap_w - gap) / min_gap_w;
        double lubforce = coeff * (factor * factor / REP_STF0_W +
                                   overlap / (min_gap_w * REP_STF1_W));
        fx += lubforce * nx;
        fy += lubforce * ny;
        fz += lubforce * nz;
    };

    double x = ppos_x[p];
    double y = ppos_y[p];
    double z = ppos_z[p];

    apply_wall_force(x - radius, 1.0, 0.0, 0.0);                                   // Left wall
    apply_wall_force((static_cast<double>(LX) - x) - radius, -1.0, 0.0, 0.0);      // Right wall
    apply_wall_force(y - radius, 0.0, 1.0, 0.0);                                   // Bottom wall
    apply_wall_force((static_cast<double>(LY) - y) - radius, 0.0, -1.0, 0.0);      // Top wall
    apply_wall_force(z - radius, 0.0, 0.0, 1.0);                                   // Front wall
    apply_wall_force((static_cast<double>(LZ) - z) - radius, 0.0, 0.0, -1.0);      // Back wall

    force_x[p] += fx;
    force_y[p] += fy;
    force_z[p] += fz;
}

} // namespace

void prepare_particle_forces_step() {
    if (!ACTIVATE_PARTICLES || NPART == 0) return;
    const size_t bytes = static_cast<size_t>(NPART) * sizeof(double);

    CHECK_CUDA_ERROR(cudaMemcpy(d_particle_force_prev2_x, d_particle_force_prev_x,
                                bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_particle_force_prev2_y, d_particle_force_prev_y,
                                bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_particle_force_prev2_z, d_particle_force_prev_z,
                                bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_particle_torque_prev2_x,  d_particle_torque_prev_x,
                                bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_particle_torque_prev2_y,  d_particle_torque_prev_y,
                                bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_particle_torque_prev2_z,  d_particle_torque_prev_z,
                                bytes, cudaMemcpyDeviceToDevice));

    CHECK_CUDA_ERROR(cudaMemcpy(d_particle_force_prev_x,  d_particle_force_accum_x,
                                bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_particle_force_prev_y,  d_particle_force_accum_y,
                                bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_particle_force_prev_z,  d_particle_force_accum_z,
                                bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_particle_torque_prev_x,   d_particle_torque_accum_x,
                                bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_particle_torque_prev_y,   d_particle_torque_accum_y,
                                bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_particle_torque_prev_z,   d_particle_torque_accum_z,
                                bytes, cudaMemcpyDeviceToDevice));

    CHECK_CUDA_ERROR(cudaMemset(d_particle_force_accum_x, 0, bytes));
    CHECK_CUDA_ERROR(cudaMemset(d_particle_force_accum_y, 0, bytes));
    CHECK_CUDA_ERROR(cudaMemset(d_particle_force_accum_z, 0, bytes));
    CHECK_CUDA_ERROR(cudaMemset(d_particle_torque_accum_x,  0, bytes));
    CHECK_CUDA_ERROR(cudaMemset(d_particle_torque_accum_y,  0, bytes));
    CHECK_CUDA_ERROR(cudaMemset(d_particle_torque_accum_z,  0, bytes));
}

void apply_repulsive_forces() {
    if (!ACTIVATE_PARTICLES || NPART == 0) return;

    const double radius = particle_radius;
    const double coeff = particle_mass * REPFORCE_SCALE;
    const double min_gap = REP_MIN_GAP;
    const double min_gap_w = REP_MIN_GAP_W;
    const int np = NPART;

    // Pairwise repulsion (soft-sphere model).
    if (np > 1) {
        dim3 block(16, 16, 1);
        dim3 grid((np + block.x - 1) / block.x, (np + block.y - 1) / block.y, 1);
        repulsive_pair_kernel<<<grid, block>>>(d_ppos_x, d_ppos_y, d_ppos_z,
                                               d_particle_force_accum_x,
                                               d_particle_force_accum_y,
                                               d_particle_force_accum_z,
                                               np, radius, coeff, min_gap);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }

    // Particle-wall repulsion.
    const int threads = 128;
    const int blocks = (np + threads - 1) / threads;
    repulsive_wall_kernel<<<blocks, threads>>>(d_ppos_x, d_ppos_y, d_ppos_z,
                                               d_particle_force_accum_x,
                                               d_particle_force_accum_y,
                                               d_particle_force_accum_z,
                                               np, radius, coeff, min_gap_w);
    CHECK_CUDA_ERROR(cudaGetLastError());
}
