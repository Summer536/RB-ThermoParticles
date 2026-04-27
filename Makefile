NVCC = nvcc
# NVCCFLAGS = -O5 -arch=sm_86 -std=c++14 -diag-suppress 20044
NVCCFLAGS = -O3 -arch=sm_86 -std=c++14 -rdc=true
INCLUDES = -Iinclude -I.

SRCS = main.cu \
	src_lbm/globals.cu \
	src_lbm/initial.cu \
	src_lbm/collision.cu \
	src_lbm/streaming.cu \
	src_lbm/macrovar.cu \
	src_pipe/globals_pipe.cu \
	src_pipe/pipe_links.cu \
	src_pipe/pipe_ibb.cu \
	output/output_flow.cu \
	output/output_statis.cu
OBJS = $(SRCS:.cu=.o)
TARGET = TC3D
HDRS = include/lbm.h include/pipe.h parameters.h

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(OBJS) -o $(TARGET)
	rm -f $(OBJS)

%.o: %.cu $(HDRS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

run: $(TARGET)
	./$(TARGET) 
