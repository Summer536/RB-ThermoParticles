NVCC = nvcc
# NVCCFLAGS = -O5 -arch=sm_86 -std=c++14 -diag-suppress 20044
NVCCFLAGS = -O3 -arch=sm_80 -std=c++14 -rdc=true
INCLUDES = -Iinclude -I.

# SRCS = main.cu globals.cu initial.cu collision.cu streaming_old.cu macrovar.cu output.cu 
SRCS = main.cu \
	src_lbm/globals.cu \
	src_lbm/initial.cu \
	src_lbm/collision.cu \
	src_lbm/streaming.cu \
	src_lbm/macrovar.cu \
	src_part/pinit.cu \
	src_part/links.cu \
	src_part/refill.cu \
	src_part/ibb.cu \
	src_part/pforces.cu \
	src_part/move.cu \
	output/output_flow.cu \
	output/output_statis.cu \
	output/output_part.cu
OBJS = $(SRCS:.cu=.o)
TARGET = rb3d
HDRS = include/lbm.h include/particle.h parameters.h

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
