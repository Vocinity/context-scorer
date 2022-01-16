message("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
message("Current config parameters are: $$CONFIG")
!build_pass{
    message("- - - - - - - - - - - - - - - Deployment - - - - - - - - - - - - - - - ")
    message('DEPS_ROOT - Default is /opt/local. You can also set it as an environment variable. As a QMake argument: qmake /path/to/ProFile.pro "DEPS_ROOT=/usr/local"')
    message('DEPL_ROOT - Default is /opt/local. You can also set it as an environment variable. As a QMake argument: qmake /path/to/ProFile.pro "DEPL_ROOT=/usr/local"')
    message('FAKE_INSTALL - Default is empty and disabled. You can also set it as an environment variable. As a QMake argument: qmake /path/to/ProFile.pro "FAKE_INSTALL=/usr/local"')
    message('PKG_BUILD_ROOT - Default is empty and disabled. You can also set it as an environment variable. As a QMake argument: qmake /path/to/ProFile.pro "PKG_BUILD_ROOT=/usr/local"')
    message("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
    message("You can pass following arguments to qmake by qmake CONFIG+=PARAMETER notion.")
    message("- - - - - - - - - - - - - - - Project Structure - - - - - - - - - - - - - - - ")
    message("TEST_OFF - all tests.")
    message("EXAMPLE_OFF - Do not build example program.")
    message("SERVER_OFF - Do not build server.")
    message("- - - - - - - - - - - - - - - CXX - - - - - - - - - - - - - - - ")
    message("CPP17 - otherwise CPP20 flags are set.")
    message("GCC8 - otherwise default gcc wil be used.")
    message("GCC9 - otherwise default gcc wil be used.")
    message("GCC10 - otherwise default gcc wil be used.")
    message("GCC11 - otherwise default gcc wil be used.")
    message("CCACHE_OFF - using this switch is disabling ccache usage.")
    message("LTO_OFF - using this switch is disabling lto usage.")
    message("GCC_CHECK_OFF - disable querying gcc version for supported c++ standard if you have a weird configuration that old gcc is overriding the new one.")
    message("- - - - - - - - - - - - - - - HW Acceleration - - - - - - - - - - - - - - - ")
    message("ONNX_OFF - Otherwise ONNX runtime expected to be found in DEPS_ROOT/include/onnx and DEPS_ROOT/lib/onnx")
    message("CUDA_OFF - otherwise CUDA is on if either /usr/local/cuda/version.txt or /usr/local/cuda/version.json is found.")
    message("USE_TORCH_CUDA_RT - this option makes use of libcudart in libtorch library path instead of the one that comes with CUDA SDK")
    message("CL_OFF - OpenCL has dependencies, information will be printed about them.")
    message("- - - - - - - - - - - - - - - https://conan.io/center - - - - - - - - - - - - - - - ")
    message("NO_CONAN - uses (shipped) conanfile.txt or conanfile.py recipes in $$PWD/../conan/ location. Empty recipes are also disabling Conan implicitly.")
    message("- - - - - - - - - - - - - - - Extensions - - - - - - - - - - - - - - - ")
    message("PYSTRING_OFF")
    message("DOUBLE_METAPHONE_OFF")
    message("SOUNDEX_OFF")
    message("RAPIDFUZZ_CPP_OFF")
    message("LEVENSHTEIN_SSE_OFF")
    message("- - - - - - - - - - - - - - - Common Compiler Options - - - - - - - - - - - - - - - ")
    message("NO_AVX_PLEASE")
    message("NO_AVX2_PLEASE")
    message("NO_BMI_PLEASE - disabling -mbmi -mbmi2 sets.")
    message("- - - - - - - - - - - - - - - ineffective for cross.pri case - - - - - - - - - - - - - - - ")
    message("AUTO_DESKTOP - is default mode. Queries your cpu against INTEL_I_SERIES__RYZEN instructions and enables found ones. Defining proper NO_AVX_PLEASE NO_AVX2_PLEASE NO_BMI_PLEASE if it is the case.")
    message("CELERON - means: -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2")
    message("ARM_DEVICE - does not set SIMD instructions. NEON-like flags are coming from Qt mkspecs.")
    message("XEON - means: -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mavx -mfma")
    message("INTEL_I_SERIES__RYZEN - means: -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mavx -mavx2 -mfma -mbmi -mbmi2")
}
message("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")

!build_pass{
    CONFIG(release, debug|release) {
        system(rm -r $${OUT_PWD}/.qmake.super)
        system(touch $${OUT_PWD}/.qmake.super)
    }

    CONFIG(debug, debug|release) {
        system(rm -r $${OUT_PWD}/.qmake.super)
        system(touch $${OUT_PWD}/.qmake.super)
    }
}

defineTest(RegisterVariable) {
    $$1{
    $$2=1
    message("$${2} is registered across submodules.")
    cache($${2},set super)
    }
}

variables+=\
CPP17\
GCC8\
GCC9\
GCC10\
GCC11\
ONNX_OFF\
CUDNN_OFF\
PYSTRING_OFF\
LEVENSHTEIN_SSE_OFF\
DOUBLE_METAPHONE_OFF\
SOUNDEX_OFF\
RAPIDFUZZ_CPP_OFF\
CPP_SUBPROCESS_OFF\
LTO_OFF\
CCACHE_OFF\
USE_TORCH_CUDA_RT\
GCC_CHECK_OFF\
DEPS_ROOT\
DEPL_ROOT\
FAKE_INSTALL \
PKG_BUILD_ROOT \
GPL\
CENTOS\
GST_PLUGIN_OFF\
EXAMPLE_OFF\
SERVER_OFF\
CUDA_OFF\
CL_OFF\
NO_CONAN\
OCV_OFF\
TORCH_OFF\
QT_OFF\
MAGIC_ENUM_OFF\
JSON_OFF\
RANGE_V3_OFF\
NUMCPP_OFF\
PARALLEL_HASHMAP_OFF\
MIO_OFF\
THRUST_OFF\
NAMED_OPERATOR_OFF\
NANO_SIGNAL_SLOT_OFF\
FAST_CPP_CSV_PARSER_OFF\
CPPITERTOOLS_OFF\
PYBIND11_OFF\
NO_AVX_PLEASE\
NO_AVX2_PLEASE\
NO_BMI_PLEASE\
CELERON\
ARM_DEVICE\
XEON\
INTEL_I_SERIES__RYZEN\
AUTO_DESKTOP

for(variable,variables){
    RegisterVariable($$variable,$$variable)
}
