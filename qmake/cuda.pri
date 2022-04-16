SYSTEM_NAME = unix         # Depending on your system either 'Win32', 'x64', or 'Win64'
SYSTEM_TYPE = 64
############################################################################################
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
############################################################################################
DEFINES +=
CONFIG +=
CUDA_DEFINES +=
DEFINES +=
CONFIG +=
CUDA_DEFINES +=

for(_defines, CUDA_DEFINES):{
    formatted_defines += -D$$_defines
}

CUDA_DEFINES = $$formatted_defines
############################################################################################
INCLUDEPATH += $${DEPS_ROOT}/include
INCLUDEPATH += $${DEPS_ROOT}/include/akil

CUDA_SDK= /usr/local/cuda/
QMAKE_LIBDIR += $$CUDA_SDK/lib64/
LIBS+= -lcufft -lcublas -lcublasLt -lcurand -lcusolver
# No spaces in path names
CUDA_SOURCES+=

INCLUDEPATH+=$${OUT_PWD}/
CUDA_OBJECTS_DIR = ${OBJECTS_DIR}
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')
CUDA_LIBS += $$join(LIBS,'.so ', '', '.so')
############################################################################################
CUDA_OBJ_FILE_PREFIX=cuda_

FORCELINK_CU_FILE = dummy.cu
forcelinkcu.target = $$FORCELINK_CU_FILE
forcelinkcu.depends = FORCE
forcelinkcu.commands = touch $$FORCELINK_CU_FILE
QMAKE_EXTRA_TARGETS += forcelinkcu
PRE_TARGETDEPS+=$$forcelinkcu.target
CUDA_SOURCES += $$FORCELINK_CU_FILE
!build_pass : write_file($$FORCELINK_CU_FILE)
var = $$basename(FORCELINK_CU_FILE)
var = $$replace(var,"\\.cu",$${QMAKE_EXT_OBJ})
CUDA_DEVICE_LINK_TRIGGER_OBJ = $${CUDA_OBJECTS_DIR}/$${CUDA_OBJ_FILE_PREFIX}$$var

NVCC_OPTIONS+= -std=c++17 --use_fast_math --ptxas-options=-v \
--expt-extended-lambda --expt-relaxed-constexpr
CONFIG(debug, debug|release) {
    cudad.input = CUDA_SOURCES
    #cudad.CONFIG += no_link
    cudad.output = $$CUDA_OBJECTS_DIR/$${CUDA_OBJ_FILE_PREFIX}${QMAKE_FILE_BASE}.o
    cudad.commands = $$CUDA_SDK/bin/nvcc -D_DEBUG $$CUDA_DEFINES --machine \
    $$SYSTEM_TYPE $$CUDA_ARCH -Xcompiler -O0 --debug -G -Xcompiler '-fPIC' $$NVCC_OPTIONS \
    -dc $$CUDA_INC $$CUDA_LIBS -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}\
            2>&1 | sed -r \"s/\\(([0-9]+)\\)/:\\1/g\" 1>&2
    cudad.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cudad

    build_pass|isEmpty(BUILDS):cudacleand.depends = compiler_cudad_clean
    else:cudacleand.CONFIG += recursive
    QMAKE_EXTRA_TARGETS += cudacleand
}
else {
    cuda.input = CUDA_SOURCES
    #cuda.CONFIG += no_link
    cuda.output = $$CUDA_OBJECTS_DIR/$${CUDA_OBJ_FILE_PREFIX}${QMAKE_FILE_BASE}.o
    cuda.commands = $$CUDA_SDK/bin/nvcc $$CUDA_DEFINES  --machine $$SYSTEM_TYPE\
    $$CUDA_ARCH -Xcompiler -O3 -Xcompiler '-fPIC' $$NVCC_OPTIONS\
    -dc $$CUDA_INC $$CUDA_LIBS -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME} \
            2>&1 | sed -r \"s/\\(([0-9]+)\\)/:\\1/g\" 1>&2
    cuda.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda

    build_pass|isEmpty(BUILDS):cudaclean.depends = compiler_cuda_clean
    else:cudaclean.CONFIG += recursive
    QMAKE_EXTRA_TARGETS += cudaclean
}

for(var, CUDA_SOURCES) {
    var = $$basename(var)
    var = $$replace(var,"\\.cu",$${QMAKE_EXT_OBJ})
    CUDEP += $${CUDA_OBJECTS_DIR}/$${CUDA_OBJ_FILE_PREFIX}$$var
}
cu_devlink.output = $$CUDA_OBJECTS_DIR/$${TARGET}_nvcc_device_compound_obj$${QMAKE_EXT_OBJ}
cu_devlink.input = CUDA_DEVICE_LINK_TRIGGER_OBJ
cu_devlink.depends = $$CUDEP
cu_devlink.dependency_type = TYPE_C
cu_devlink.commands = $$CUDA_SDK/bin/nvcc --machine $$SYSTEM_TYPE \
    $$CUDA_ARCH $$CUDA_DEFINES $$NVCC_OPTIONS -Xcompiler '-fPIC' \
    -dlink $$CUDEP \
    -o $$CUDA_OBJECTS_DIR/$${TARGET}_nvcc_device_compound_obj$${QMAKE_EXT_OBJ}
cu_devlink.name = cuda_device_linker
#cu_devlink.variable_out = OBJECTS
cu_devlink.CONFIG += target_predeps
QMAKE_EXTRA_COMPILERS += cu_devlink
#QMAKE_PRE_LINK += $${cu_devlink.commands}
