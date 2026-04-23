#ifndef PARAMETERS_H
#define PARAMETERS_H

// Continue calculation
#define CONTINUOUS_STEPS 0

// Grid dimensions
#define LX 216
#define LY 216
#define LZ 32
#define LXY (LX * LY)
#define LXYZ (LX * LY * LZ)
#define NPOP 27

// Time steps
#define NEND 5000
#define NDIAG 125
#define NFLOWOUT 5000
#define NNUOUT 125
#define NOUT_P 125

// Physical parameters
#define RAYLEIGH 1.0e7
#define PRANDTL 0.71
#define T_HOT 1.0
#define T_COLD 0.0
#define T_REF 0.5
#define BETA 0.0000625
#define VISC 0.01206
#define RHO0 1.0

// Particle controls
#define ACTIVATE_PARTICLES 1
#define PARTICLE_FRACTION 0.05
#define PARTICLE_DIAM_FRAC (1.0/20.0)
#define CP_PARTICLE_RATIO 10.0
#define CP_FLUID 1.0
#define RHO_PARTICLE_RATIO 1.0
#define REPFORCE_SCALE 2.0e-4
#define PARTICLE_INIT_TEMP (T_COLD + T_HOT)/2

#define PARTICLE_RAND_SEED 86143 
#define REP_MIN_GAP 2.0
#define REP_MIN_GAP_W 2.0
#define REP_STF0 0.025
#define REP_STF1 0.002
#define REP_STF0_W 0.025
#define REP_STF1_W 0.002

// PIPE parameters
#define ACTIVATE_PIPE 1
#define PIPE_RAD1 100.0
#define PIPE_RAD2 50.0
#define PIPE_u0 0.05

// Block dimensions for CUDA kernels
#define BLOCK_X 8
#define BLOCK_Y 8
#define BLOCK_Z 4

#endif 
