SYSTEM_NAME = unix         # Depending on your system either 'Win32', 'x64', or 'Win64'
SYSTEM_TYPE = 64

DEFINES +=
CONFIG +=
CUDA_DEFINES +=
DEFINES +=
CONFIG +=
CUDA_DEFINES +=

for(_defines, CUDA_DEFINES):{
    formatted_defines += -D$$_defines
}

formatted_defines += -std=c++14 -expt-relaxed-constexpr
CUDA_DEFINES = $$formatted_defines

!isEmpty(CUDA_WANTED_ARCHS){
    CUDA_COMPUTE_ARCH += $$CUDA_WANTED_ARCHS
}
else{
    system("chmod +x $$PWD/detectCudaArch.sh")
    CUDA_COMPUTE_ARCH+=$$system("$$PWD/detectCudaArch.sh")
}
CUDA_SORTED_ARCHS=$$sorted(CUDA_COMPUTE_ARCH)
CUDA_MINIMUM_ARCH=$$first(CUDA_SORTED_ARCHS)
for(_a, CUDA_COMPUTE_ARCH):{
    formatted_arch =$$join(_a,'',' -gencode arch=compute_',',code=sm_$$_a')
    CUDA_ARCH += $$formatted_arch
    message("Compiling for CUDA arch:" $$_a)
}
greaterThan(CUDA_MINIMUM_ARCH, 52) {
    !equals(CUDA_MINIMUM_ARCH, 61) {
        message("CUDA FP16 support is available")
        DEFINES +=CUDA_FP16_AVAILABLE
        CONFIG +=CUDA_FP16_AVAILABLE
    }
}
greaterThan(CUDA_MINIMUM_ARCH, 60) {
    !equals(CUDA_MINIMUM_ARCH, 62) {
        message("CUDA INT8 support is available")
        DEFINES +=CUDA_INT8_AVAILABLE
        CONFIG +=CUDA_INT8_AVAILABLE
    }
}
greaterThan(CUDA_MINIMUM_ARCH, 62) {
    message("CUDA FP16 TENSOR CORES support is available")
    DEFINES +=CUDA_FP16_TENSOR_CORES_AVAILABLE
    CONFIG +=CUDA_FP16_TENSOR_CORES_AVAILABLE
}
greaterThan(CUDA_MINIMUM_ARCH, 70) {
    message("CUDA INT8 TENSOR CORES support is available")
    DEFINES +=CUDA_INT8_TENSOR_CORES_AVAILABLE
    CONFIG +=CUDA_INT8_TENSOR_CORES_AVAILABLE
}
equals(CUDA_MINIMUM_ARCH, 72) {
    message("CUDA DLA support is available")
    DEFINES +=CUDA_DLA_AVAILABLE
    CONFIG +=CUDA_DLA_AVAILABLE
}

CUDA_SDK= /usr/local/cuda/
!USE_TORCH_CUDA_RT{
    QMAKE_LIBDIR += $$CUDA_SDK/lib64/
    LIBS+= -lcufft -lcublas -lcurand -lcusolver
}
CUDA_OBJECTS_DIR = ${OBJECTS_DIR}
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')
CUDA_LIBS += $$join(LIBS,'.so ', '', '.so')

# No spaces in path names
LIGHTSEQ_AVAILABLE{
    CUDA_SOURCES+= $$PWD/../3rdparty/lightseq/lightseq/inference/model/gpt_encoder.cc.cu \
                   $$PWD/../3rdparty/lightseq/lightseq/inference/tools/util.cc.cu \
                   $$PWD/../3rdparty/lightseq/lightseq/inference/kernels/gptKernels.cc.cu
}

NVCC_OPTIONS += --use_fast_math --ptxas-options=-v
CONFIG(debug, debug|release) {
        cuda_d.input = CUDA_SOURCES
        cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
        cuda_d.commands = $$CUDA_SDK/bin/nvcc -D_DEBUG $$CUDA_DEFINES --machine \
        $$SYSTEM_TYPE $$CUDA_ARCH -Xcompiler '-fPIC' -dc $$NVCC_OPTIONS $$CUDA_INC\
        $$CUDA_LIBS -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}\
                2>&1 | sed -r \"s/\\(([0-9]+)\\)/:\\1/g\" 1>&2
        cuda_d.dependency_type = TYPE_C
        QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
        cuda.input = CUDA_SOURCES
        cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
        cuda.commands = $$CUDA_SDK/bin/nvcc $$CUDA_DEFINES  --machine $$SYSTEM_TYPE\
        $$CUDA_ARCH -Xcompiler '-fPIC' -dc $$NVCC_OPTIONS $$CUDA_INC $$CUDA_LIBS -o \
        ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME} \
                2>&1 | sed -r \"s/\\(([0-9]+)\\)/:\\1/g\" 1>&2
        cuda.dependency_type = TYPE_C
        QMAKE_EXTRA_COMPILERS += cuda
}
