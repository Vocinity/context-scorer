# Context Scorer
###### Author: [@isgursoy](https://www.github.com/isgursoy)
###### **BEWARE**: You have to donate $5 to [charities](https://donatenow.wfp.org/wfp/~my-donation) for each Batman call without properly following readme.

Repository contains lib_Context_Scorer, and Context_Scorer (example) executable which is demonstrating context scoring.

You don't have time for details? Jump right to the [example building procedure.](https://github.com/Vocinity/context-scorer#example-building-procedure)

## Documentation
Doxygen Doxyfile is provided in doc folder but you can live without it. lib_Context_Scorer code is documenting itself.

## Python Environment

An environment which provides pip/requirements.txt packages. Python environment is totally irrelevant for cpp runtime
and vice versa. After this point, only cpp environment is documented in this page.

## C++ Environment

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/std/the-standard)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://isocpp.org/std/the-standard)
[![C++23](https://img.shields.io/badge/C%2B%2B-23-blue.svg)](https://isocpp.org/std/the-standard)
 Feature Map: https://en.cppreference.com/w/cpp/feature_test
  
Visual Studio: NO  
[GNU](https://gcc.gnu.org/onlinedocs/libstdc++/manual/status.html#status.iso.2020): 10.1+  
 Clang: 12+ (untested and not maintaining but it is trivial to port)  

### Dependencies

#### Context_Scorer
- lib_Context-Scorer 

#### lib_Context-Scorer
- [aMisc](https://github.com/Vocinity/aMisc)
  - TORCH *is mandatory.*
  - MAGIC_ENUM *is a must.*
  - RANGE_V3 *is optional for pre-C++20. (If you suffer from [compiler seg fault](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=96720), disable RANGE_V3 and use C++20)*
  - JSON *is a must.*
  - PYSTRING or RANGE_V3 or c++23 support *is needed.*
  - LEVENSHTEIN_SSE or RAPIDFUZZ_CPP *or both would be nice.*
  - SOUNDEX *is optional.*
  - DOUBLE_METAPHONE *is optional.*

### Building
Did you install aMisc? Congrats, then already you did almost all the job.

Build system is qmake. As like other make preprocessors, qmake is bringing you to the make step by
taking care of Makefile customization instructions.

Here typical qmake based build-install procedure:
```bash
cd builDir
qmake path/to/root/pro/file.pro CONFIG+=THIS_IS_A_SWITCH "ARGUMENT=THIS_IS_A_VALUE"
make
make install
```
#### Customization

*Running `qmake` command will show customization information alongside available parameters and how to use them.*

> Common  and lib_Context_Scorer

| **Argument**                | **Default Value**  |**Notes**                         |
|:---------------------------:|:------------------:|:--------------------------------:|
| ```DEPS_ROOT```             | ```/opt/local```   | Where to consume dependencies. You can also set it as environment variable. As a QMake argument: `qmake /path/to/ProFile.pro "DEPS_ROOT=/usr/local"` and qmake argument overrides env var. If env var is not set and qmake argument is omitted, it will be set as `/opt/local/`.
| ```DEPL_ROOT```             | ```/opt/local```   | Where to install artifacts. You can also set it as environment variable. As a QMake argument: `qmake /path/to/ProFile.pro "DEPL_ROOT=/usr/local"` and qmake argument overrides env var. If env var is not set and qmake argument is omitted, it will be set as `/opt/local/`.
| ```FAKE_INSTALL```          | ```empty```        | Way to have an independent build root for packaging without affecting `make install` procedure. Only copies runtime materials like executables and libraries without headers and inclusion helpers. You can also set it as environment variable. As a QMake argument: `qmake /path/to/ProFile.pro "FAKE_INSTALL=/usr/local"` and qmake argument overrides env var. If env var is not set and qmake argument is omitted, it will be set as `/opt/local/`.
| ```PKG_BUILD_ROOT```        | ```empty```        | If you are installing into the build root, you need to provide this argument or environment variable to be able to use inclusion helpers from real install root in runtime. As a QMake argument: `qmake /path/to/ProFile.pro "PKG_BUILD_ROOT=/usr/local"` and qmake argument overrides env var. For example you should set `PKG_BUILD_ROOT=/home/user/build_root` and `DEPL_ROOT=/opt/local`

| **Switch**                  | **Default Value**  |**By Default**                         |
|:---------------------------:|:------------------:|:-------------------------------------:|
| ```debug```                 | ```0```            | by default, release binary will be produced. Note that `debug` word is all lowercase.
| ```unversioned_libname```   | ```0```            | by default, version, datetime and commit hash will be appended to the output and symlinks will be produced. Note that `debug` word is all lowercase.
| ```CPP17```                 | ```0```            | CPP20 is used
| ```CENTOS```                | ```0```            | If 1 (so you passed), sox will be included from `/usr/include/sox` instead of ubuntu's `/usr/include/`.
| ```GCC8```                  | ```0```            | default gcc wil be used.
| ```GCC9```                  | ```0```            | /\
| ```GCC10```                 | ```0```            | /\
| ```GCC11```                 | ```0```            | /\
| ```PYSTRING_OFF```          | ```0```            | /\
| ```DOUBLE_METAPHONE_OFF```  | ```0```            | /\
| ```SOUNDEX_OFF```           | ```0```            | /\
| ```RAPIDFUZZ_CPP_OFF```     | ```0```            | /\
| ```LEVENSHTEIN_SSE_OFF```   | ```0```            | /\
| ```LTO_OFF```               | ```0```            | enabled by default. If you have a mysterious seg fault in linker, this is way to go.
| ```CCACHE_OFF```            | ```0```            | using this switch is disabling ccache usage.
| ```GCC_CHECK_OFF```         | ```0```            | C++ standard is deduced automatically from gcc version. Disable querying gcc version for supported c++ standard if you have a weird configuration that old gcc is overriding the new one and you are sure there is desired CPP17/20 support.
| ```TEST_OFF```              | ```0```            | Test (that dont exist) wont be compiled
| ```ONNX_OFF```              | ```0```            | Otherwise ONNX runtime expected to be found in DEPS_ROOT/include/onnx and DEPS_ROOT/lib/onnx.
| ```CUDA_OFF```              | ```0```            | CUDA is used if either `/usr/local/cuda/version.txt` or `/usr/local/cuda/version.json` is found. Please dont miss libtorch and installed cuda version compatibility note in [External Dependencies](https://github.com/Vocinity/aMisc#external-dependencies) section.
| ```USE_TORCH_CUDA_RT```     | ```0```            | This option makes use of libcudart in libtorch library path instead of the one that comes with CUDA SDK. Normally they can live together but some systems require to enable this switch.
| ```CL_OFF```                | ```0```            | OpenCL is enabled. See aMisc [External Dependencies](https://github.com/Vocinity/aMisc#external-dependencies). 
| ```NO_CONAN```              | ```0```            | Conan is enabled but currently not used as the result of empty `conan/conanfile.txt` recipe
| ```NO_AVX_PLEASE```         | ```0```            | AVX is enabled. Run`cat /proc/cpuinfo` to see supported CPU instructions on your target system.
| ```NO_AVX2_PLEASE```        | ```0```            | /\ 
| ```NO_BMI_PLEASE```         | ```0```            | -mbmi -mbmi2 sets are enabled.
| ```AUTO_DESKTOP```          | ```1```            | queries your cpu against INTEL_I_SERIES__RYZEN instructions and enables found ones. Defining proper NO_AVX_PLEASE NO_AVX2_PLEASE NO_BMI_PLEASE if it is the case.
| ```INTEL_I_SERIES__RYZEN``` | ```0```            | Nothing done by default, AUTO_DESKTOP is running. Enables -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mavx -mavx2 -mfma -mbmi -mbmi2 if passed.
| ```CELERON```               | ```0```            | Nothing done by default, AUTO_DESKTOP is running. Enables -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2  if passed.
| ```ARM_DEVICE```            | ```0```            | Nothing done by default, AUTO_DESKTOP is running. If you pass this switch to the qmake, then SIMD instructions are off and NEON-like flags are coming from Qt mkspecs.
| ```XEON```                  | ```0```            | Nothing done by default, AUTO_DESKTOP is running. Enables -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mavx -mfma  if passed.

> context-scorer-example

| **Switch**                  | **Default Value**  |**By Default**                              |
|:---------------------------:|:------------------:|:-------------------------------------:|
| ```EXAMPLE_OFF```           | ```0```            | Building example


#### Installation Material

> Context_Scorer

*One* of below depending relese|debug build and hw acceleration backend.
```
DEPL_ROOT/bin/akil/Context-Scorer_cu
DEPL_ROOT/bin/akil/Context-Scorer_cl
DEPL_ROOT/bin/akil/Context-Scorer_cpu
DEPL_ROOT/bin/akil/Context-Scorer_cu+dbg
DEPL_ROOT/bin/akil/Context-Scorer_cl+dbg
DEPL_ROOT/bin/akil/Context-Scorer_cpu+dbg
```
> lib_Context_Scorer

Release build of the commit 6691d7dcfeb5076db749a0cba25b48cfe5395379@branch and cpu backend would install:
```
DEPL_ROOT/lib/akil/lib_Context-Scorer_cpu.so
DEPL_ROOT/lib/akil/lib_Context-Scorer_cpu.so.2021-08-05_19:02@6691d7dcfeb5076db749a0cba25b48cfe5395379
DEPL_ROOT/lib/akil/lib_Context-Scorer_cpu.so.2021-08-05_19:02@6691d7dcfeb5076db749a0cba25b48cfe5395379.0
DEPL_ROOT/lib/akil/lib_Context-Scorer.so.2021-08-05_19:02@6691d7dcfeb5076db749a0cba25b48cfe5395379.0.0

DEPL_ROOT/include/akil/Context_Scorer.hpp

DEPL_ROOT/share/akil/qmake/depend_context_scorer.pri
```
For qmake build system, including only `depend_context_scorer.pri`is enough for everything; header, libraries, lower level dependencies...
Just like how Context_Scorer do.

> Conan

If in use, Conan dependencies and `conanbuildinfo.pri`go to `DEPL_ROOT/conan/pro-file-name-of-the-subproject`:
```
DEPL_ROOT/conan/context-scorer-example
DEPL_ROOT/conan/context-scorer-library
```
and you will find `DEPL_ROOT/share/akil/qmake/pro-file-name-of-the-subproject-conan-linker-runtime.sh`:
```
DEPL_ROOT/share/akil/qmake/context-scorer-example-conan-linker-runtime.sh
DEPL_ROOT/share/akil/qmake/context-scorer-library-conan-linker-runtime.sh
```
You should export your Conan library paths by using these scripts to your linker in case of Conan use.

> Cleaning

And to cleanup *everything* in `DEPL_ROOT` depending release|debug:
```bash
cd buildDir
make clean
make distclean
make uninstall
```

#### Example Building Procedure

> We are on Centos, default gcc is 8.4, we dont have a nvidia gpu, our cpu has not igpu.

* Get some dependencies from yum
```bash
sudo yum install sox-devel python3-devel qt5-qtbase-devel tbb-devel
```
* Torch 1.9+ is in `/opt/local/include/torch`and `/opt/local/lib/torch`. `/opt/local` is our DEPS_ROOT. [aMisc Customization](https://github.com/Vocinity/aMisc#customization) is talking about it in dependencies perspective.
```
/opt/local/include/torch
├── ATen
├── c10
├── c10d
├── caffe2
├── nvfuser_resources
├── pybind11
├── sleef.h
├── TH
├── THC
├── THCUNN
└── torch
10 directories, 1 file
```
```
/opt/local/lib/torch
├── libasmjit.a
├── libbackend_with_compiler.so
├── libbenchmark.a
├── libbenchmark_main.a
├── libc10_cuda.so
├── libc10d.a
├── libc10d_cuda_test.so
├── libc10.so
├── libcaffe2_detectron_ops_gpu.so
├── libcaffe2_module_test_dynamic.so
├── libcaffe2_nvrtc.so
├── libcaffe2_observers.so
├── libCaffe2_perfkernels_avx2.a
├── libCaffe2_perfkernels_avx512.a
├── libCaffe2_perfkernels_avx.a
├── libcaffe2_protos.a
├── libclog.a
├── libcpuinfo.a
├── libcpuinfo_internals.a
├── libcudart-6d56b25a.so.11.0
├── libcudart.so.11.0 -> libcudart-6d56b25a.so.11.0
├── libdnnl.a
├── libfbgemm.a
├── libfmt.a
├── libfoxi_loader.a
├── libgloo.a
├── libgloo_cuda.a
├── libgmock.a
├── libgmock_main.a
├── libgomp-75eea7e8.so.1
├── libgomp.so.1 -> libgomp-75eea7e8.so.1
├── libgtest.a
├── libgtest_main.a
├── libjitbackend_test.so
├── libkineto.a
├── libmkldnn.a
├── libnnpack.a
├── libnnpack_reference_layers.a
├── libnvrtc-3a20f2b6.so.11.1
├── libnvrtc-builtins-07fb3db5.so.11.1
├── libnvrtc-builtins.so.11.1 -> libnvrtc-builtins-07fb3db5.so.11.1
├── libnvrtc.so.11.1 -> libnvrtc-3a20f2b6.so.11.1
├── libnvToolsExt-24de1d56.so.1
├── libnvToolsExt.so.1 -> libnvToolsExt-24de1d56.so.1
├── libonnx.a
├── libonnx_proto.a
├── libprocess_group_agent.so
├── libprotobuf.a
├── libprotobuf-lite.a
├── libprotoc.a
├── libpthreadpool.a
├── libpytorch_qnnpack.a
├── libqnnpack.a
├── libshm.so
├── libtensorpipe.a
├── libtensorpipe_agent.so
├── libtensorpipe_uv.a
├── libtorchbind_test.so
├── libtorch_cpu.so
├── libtorch_cuda_cpp.so
├── libtorch_cuda_cu.so
├── libtorch_cuda.so
├── libtorch_global_deps.so
├── libtorch_python.so
├── libtorch.so
└── libXNNPACK.a
0 directories, 66 files
```
and linker is able to see torch libraries. Either by `export`ing `LD_LIBRARY_PATH` or ld.so.conf:
```
$ cat /etc/ld.so.conf.d/opt.conf 
/opt/local/lib/akil
/opt/local/lib/torch
```

* Prepare aMisc:
```bash
cd /home/vocinity/work
git clone --recurse-submodules https://github.com/Vocinity/aMisc
cd aMisc
mkdir build
cd build
```
* Setup gcc 10 by sourcing secondary environment:
```bash
source /opt/rh/gcc-toolset-10/enable
```

- Configure aMisc:
  - RANGE_V3_OFF because your strange [compiler](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=96720) may crash with that.
  - USE_TORCH_CUDA_RT because you are getting a mysterios cublas crash in your environment.
  - OCV_OFF because we dont need it for Noise Reduction and it is an external dependency that requires you to compile yourself.
  - CL_OFF because we dont need OpenCL availability in CUDA build.
  - NO_CONAN because we are using yum.
  - QT_OFF so we just need qmake, we do not need Qt framework libraries for Context Scorer.
  - LTO_OFF because your compiler is a dumb.

 ```bash
qmake-qt5 .. CONFIG+=RANGE_V3_OFF CONFIG+=OCV_OFF CONFIG+=USE_TORCH_CUDA_RT CONFIG+=CL_OFF CONFIG+=NO_CONAN CONFIG+=QT_OFF CONFIG+=LTO_OFF
```


- Build aMisc:
```bash
make -j 8 install
```

If you can locate [these](https://github.com/Vocinity/aMisc#installation-material) files, we are ready to compile Context Scorer.

---

* Prepare Context Scorer:
```bash
cd /home/vocinity/work
git clone https://github.com/Vocinity/context-scorer
cd context-context
mkdir build
cd build
```

- Configure Context Scorer:
  - RANGE_V3_OFF because your strange [compiler](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=96720) may crash with that.
  - USE_TORCH_CUDA_RT because you are getting a mysterios cublas crash in your environment.
  - CL_OFF because we dont need OpenCL availability in CUDA build.
  - NO_CONAN because we are using yum.
  - LTO is off because you know why.

 ```bash
qmake-qt5 .. CONFIG+=RANGE_V3_OFF CONFIG+=USE_TORCH_CUDA_RT CONFIG+=CL_OFF CONFIG+=NO_CONAN CONFIG+=LTO_OFF
```
* Build Context Scorer:
```bash
make -j 8
make install
```
* For the sake of having a complete example, obviously you need to arrange your linker paths for runtime too. Remember torch section above and here 3rd line is new:
```
$ cat /etc/ld.so.conf.d/opt.conf 
/opt/local/lib
/opt/local/lib/torch
/opt/local/lib/akil
```
(Compilation is locating libraries by known exact paths, also you should tell linker where can be your libSomething.so in runtime.)

If you can locate [these](https://github.com/Vocinity/context-scorer#installation-material) files, you did it. Have fun!

### Test
Currently there are no automated tests. You can compare output of Context_Scorer by using [models/master/context-scorer/](https://github.com/Vocinity/models/blob/master/context-scorer/) distillgpt2 cuda model.
```bash
sentence: Click on the eye in the icon tray to pick your product of interest or say echelon-connect bike or smart rower. Smart rower.
negative_log_likelihood: 385.103
production: -5.19577
mean: 0.00442321
g_mean: 0.000173123
h_mean: -0.296845
loss: 5264.53
sentence_probability: -1.76413e-169
```

## License
Vocinity Licensing Terms [@sipvoip](https://www.github.com/sipvoip)

## Diary
### - October 30th -
* Release
