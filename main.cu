#include "particle.h"
#include <cmath>

int main() {
    struct timeval start_time, end_time;
    double cpu_time_used = 0.0, gpu_time_used = 0.0;
    cudaEvent_t gpu_start, gpu_stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&gpu_start));
    CHECK_CUDA_ERROR(cudaEventCreate(&gpu_stop));
    gettimeofday(&start_time, NULL);

    initialize();

    CHECK_CUDA_ERROR(cudaMemcpy(h_rho, d_rho, LXYZ * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_ux,  d_ux,  LXYZ * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_uy,  d_uy,  LXYZ * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_uz,  d_uz,  LXYZ * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_phi, d_phi, LXYZ * sizeof(double), cudaMemcpyDeviceToHost));

    dim3 grid((LX + BLOCK_X - 1) / BLOCK_X,
              (LY + BLOCK_Y - 1) / BLOCK_Y,
              (LZ + BLOCK_Z - 1) / BLOCK_Z);
    dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);

    const int start_step = restart_step;
    const int final_step = NEND;

    if (ACTIVATE_PARTICLES && NPART > 0 && start_step == 0) {
        output_particles(0);
    }

    for (int istep = start_step + 1; istep <= final_step; istep++) {
        // GPU timing start
        CHECK_CUDA_ERROR(cudaEventRecord(gpu_start));
        macrovar<<<grid, block>>>(d_f, d_g, d_force_realx, d_force_realy,
            d_force_realz, d_rho, d_ux, d_uy, d_uz, d_phi); 
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize()); 

        forcing<<<grid, block>>>(d_force_realx, d_force_realy, d_force_realz, d_phi);

        collision_BGK<<<grid, block>>>(d_f, d_force_realx, d_force_realy, d_force_realz,
                                       d_rho, d_ux, d_uy, d_uz);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaMemcpy(d_f_collide, d_f, NPOP * LXYZ * sizeof(double), cudaMemcpyDeviceToDevice));
        
        streaming<<<grid, block>>>(d_f, d_f_temp);
        CHECK_CUDA_ERROR(cudaMemcpy(d_f, d_f_temp, NPOP * LXYZ * sizeof(double), cudaMemcpyDeviceToDevice));
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        if (ACTIVATE_PARTICLES && NPART > 0) {
            bounce_back_particles();
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        }

        collision_BGK_scalar<<<grid, block>>>(d_g, d_force_realx, d_force_realy, d_force_realz,
                                              d_rho, d_ux, d_uy, d_uz, d_phi);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize()); 
        CHECK_CUDA_ERROR(cudaMemcpy(d_g_collide, d_g, NPOP * LXYZ * sizeof(double), cudaMemcpyDeviceToDevice));

        streaming_scalar<<<grid, block>>>(d_g, d_g_temp, d_rho);
        CHECK_CUDA_ERROR(cudaMemcpy(d_g, d_g_temp, NPOP * LXYZ * sizeof(double), cudaMemcpyDeviceToDevice));
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        if (ACTIVATE_PARTICLES && NPART > 0) {
            bounce_back_thermal_particles();
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        }

        
        if (ACTIVATE_PARTICLES && NPART > 0) {
            prepare_particle_forces_step();
            apply_repulsive_forces();

            compute_particle_forces();
            compute_particle_heat();

            update_particles();

            build_links();
            refill_nodes();
        }

        // GPU timing end
        CHECK_CUDA_ERROR(cudaEventRecord(gpu_stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(gpu_stop));

        float gpu_milliseconds = 0.0f;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpu_milliseconds, gpu_start, gpu_stop));
        gpu_time_used += (double)gpu_milliseconds / 1000.0;

        // CPU timing start
        clock_t cpu_start = clock();

        if (istep % NDIAG == 0) {
            diag_flow(istep);
        }
        if (istep % NFLOWOUT == 0 || istep == final_step) {
            output_flow(istep);
        }
        if (istep % NOUT_P == 0) {
            output_particles(istep);
        }
        if (istep % NNUOUT == 0) {
            output_nu(istep);
            output_profile(istep);
        }
        if (istep == final_step) {
            output_fg(istep);
            output_particles(istep);
        }

        // CPU timing end
        cpu_time_used += (double)(clock() - cpu_start) / CLOCKS_PER_SEC;
    }

    // calculate total time
    gettimeofday(&end_time, NULL);
    double total_time = (end_time.tv_sec - start_time.tv_sec) + 
                       (end_time.tv_usec - start_time.tv_usec) / 1000000.0;

    printf("Simulation completed.\n");
    printf("Rayleigh number: %.2e\n", rayl);
    printf("Prandtl number: %.2f\n", prand);
    printf("Total iterations: %d\n", NEND);
    if (start_step > 0) {
        printf("Restarted from step: %d, final step: %d\n", start_step, final_step);
    } else {
        printf("Final step: %d\n", final_step);
    }
    printf("CPU computation time: %.2f seconds\n", cpu_time_used);
    printf("GPU computation time: %.2f seconds\n", gpu_time_used);
    printf("Total computation time: %.2f seconds\n", total_time);

    CHECK_CUDA_ERROR(cudaEventDestroy(gpu_start));
    CHECK_CUDA_ERROR(cudaEventDestroy(gpu_stop));

    CHECK_CUDA_ERROR(cudaFree(d_f));
    CHECK_CUDA_ERROR(cudaFree(d_g));
    CHECK_CUDA_ERROR(cudaFree(d_rho));
    CHECK_CUDA_ERROR(cudaFree(d_ux));
    CHECK_CUDA_ERROR(cudaFree(d_uy));
    CHECK_CUDA_ERROR(cudaFree(d_uz));
    CHECK_CUDA_ERROR(cudaFree(d_phi));
    CHECK_CUDA_ERROR(cudaFree(d_force_realx));
    CHECK_CUDA_ERROR(cudaFree(d_force_realy));
    CHECK_CUDA_ERROR(cudaFree(d_force_realz));
    CHECK_CUDA_ERROR(cudaFree(d_f_temp));
    CHECK_CUDA_ERROR(cudaFree(d_f_collide));
    CHECK_CUDA_ERROR(cudaFree(d_g_temp));
    CHECK_CUDA_ERROR(cudaFree(d_g_collide));
    CHECK_CUDA_ERROR(cudaFree(d_ted));
    CHECK_CUDA_ERROR(cudaFree(d_Ked));
    CHECK_CUDA_ERROR(cudaFree(d_ppos_x));
    CHECK_CUDA_ERROR(cudaFree(d_ppos_y));
    CHECK_CUDA_ERROR(cudaFree(d_ppos_z));
    CHECK_CUDA_ERROR(cudaFree(d_pvel_x));
    CHECK_CUDA_ERROR(cudaFree(d_pvel_y));
    CHECK_CUDA_ERROR(cudaFree(d_pvel_z));
    CHECK_CUDA_ERROR(cudaFree(d_pomega_x));
    CHECK_CUDA_ERROR(cudaFree(d_pomega_y));
    CHECK_CUDA_ERROR(cudaFree(d_pomega_z));
    CHECK_CUDA_ERROR(cudaFree(d_ptheta_x));
    CHECK_CUDA_ERROR(cudaFree(d_ptheta_y));
    CHECK_CUDA_ERROR(cudaFree(d_ptheta_z));
    CHECK_CUDA_ERROR(cudaFree(d_pforce_x));
    CHECK_CUDA_ERROR(cudaFree(d_pforce_y));
    CHECK_CUDA_ERROR(cudaFree(d_pforce_z));
    CHECK_CUDA_ERROR(cudaFree(d_ptorque_x));
    CHECK_CUDA_ERROR(cudaFree(d_ptorque_y));
    CHECK_CUDA_ERROR(cudaFree(d_ptorque_z));
    CHECK_CUDA_ERROR(cudaFree(d_pforce_hist_x));
    CHECK_CUDA_ERROR(cudaFree(d_pforce_hist_y));
    CHECK_CUDA_ERROR(cudaFree(d_pforce_hist_z));
    CHECK_CUDA_ERROR(cudaFree(d_ptorque_hist_x));
    CHECK_CUDA_ERROR(cudaFree(d_ptorque_hist_y));
    CHECK_CUDA_ERROR(cudaFree(d_ptorque_hist_z));
    CHECK_CUDA_ERROR(cudaFree(d_ptemp));
    CHECK_CUDA_ERROR(cudaFree(d_pheat));
    CHECK_CUDA_ERROR(cudaFree(d_pheat_prev));
    CHECK_CUDA_ERROR(cudaFree(d_particle_force_accum_x));
    CHECK_CUDA_ERROR(cudaFree(d_particle_force_accum_y));
    CHECK_CUDA_ERROR(cudaFree(d_particle_force_accum_z));
    CHECK_CUDA_ERROR(cudaFree(d_particle_torque_accum_x));
    CHECK_CUDA_ERROR(cudaFree(d_particle_torque_accum_y));
    CHECK_CUDA_ERROR(cudaFree(d_particle_torque_accum_z));
    CHECK_CUDA_ERROR(cudaFree(d_particle_force_prev_x));
    CHECK_CUDA_ERROR(cudaFree(d_particle_force_prev_y));
    CHECK_CUDA_ERROR(cudaFree(d_particle_force_prev_z));
    CHECK_CUDA_ERROR(cudaFree(d_particle_force_prev2_x));
    CHECK_CUDA_ERROR(cudaFree(d_particle_force_prev2_y));
    CHECK_CUDA_ERROR(cudaFree(d_particle_force_prev2_z));
    CHECK_CUDA_ERROR(cudaFree(d_particle_torque_prev_x));
    CHECK_CUDA_ERROR(cudaFree(d_particle_torque_prev_y));
    CHECK_CUDA_ERROR(cudaFree(d_particle_torque_prev_z));
    CHECK_CUDA_ERROR(cudaFree(d_particle_torque_prev2_x));
    CHECK_CUDA_ERROR(cudaFree(d_particle_torque_prev2_y));
    CHECK_CUDA_ERROR(cudaFree(d_particle_torque_prev2_z));
    CHECK_CUDA_ERROR(cudaFree(d_ibnode));
    CHECK_CUDA_ERROR(cudaFree(d_ibnode_prev));
    CHECK_CUDA_ERROR(cudaFree(d_ibnode_owner));
    CHECK_CUDA_ERROR(cudaFree(d_ibnode_owner_prev));
    if (d_link_count) {
        CHECK_CUDA_ERROR(cudaFree(d_link_count));
    }
    if (d_link_offset) {
        CHECK_CUDA_ERROR(cudaFree(d_link_offset));
    }
    if (d_link_scan_tmp) {
        CHECK_CUDA_ERROR(cudaFree(d_link_scan_tmp));
    }
    CHECK_CUDA_ERROR(cudaFree(d_new_fluid_nodes));
    CHECK_CUDA_ERROR(cudaFree(d_new_fluid_pids));
    CHECK_CUDA_ERROR(cudaFree(d_particle_links));

    free(h_f);
    free(h_g);
    free(h_rho);
    free(h_ux);
    free(h_uy);
    free(h_uz);
    free(h_phi);
    free(h_ted);
    free(h_Ked);
    free(h_ppos_x);
    free(h_ppos_y);
    free(h_ppos_z);
    free(h_pvel_x);
    free(h_pvel_y);
    free(h_pvel_z);
    free(h_pomega_x);
    free(h_pomega_y);
    free(h_pomega_z);
    free(h_ptheta_x);
    free(h_ptheta_y);
    free(h_ptheta_z);
    free(h_pforce_x);
    free(h_pforce_y);
    free(h_pforce_z);
    free(h_ptorque_x);
    free(h_ptorque_y);
    free(h_ptorque_z);
    free(h_pforce_hist_x);
    free(h_pforce_hist_y);
    free(h_pforce_hist_z);
    free(h_ptorque_hist_x);
    free(h_ptorque_hist_y);
    free(h_ptorque_hist_z);
    free(h_ptemp);
    free(h_pheat);
    free(h_pheat_prev);
    free(h_particle_force_accum_x);
    free(h_particle_force_accum_y);
    free(h_particle_force_accum_z);
    free(h_particle_torque_accum_x);
    free(h_particle_torque_accum_y);
    free(h_particle_torque_accum_z);
    free(h_ibnode);
    free(h_ibnode_prev);
    free(h_ibnode_owner);
    free(h_ibnode_owner_prev);
    free(h_new_fluid_nodes);
    free(h_new_fluid_pids);

    CHECK_CUDA_ERROR(cudaDeviceReset());

    return 0;
}
