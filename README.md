# Context Scorer
###### Core: [@isgursoy](https://www.github.com/isgursoy), Server: [@sind4l](https://github.com/sind4l)
###### **BEWARE**: You have to donate $5 to [charities](https://donatenow.wfp.org/wfp/~my-donation) for each Batman call without properly following readme.

Repository contains lib_Context-Scorer, Context-Scorer (example) executable which is demonstrating context scoring and Context-Score_Server (grpc service).

You don't have time for details? Jump right to the [example building procedure.](https://github.com/Vocinity/context-scorer#build-procedure)

## Documentation
Doxygen Doxyfile is provided in doc folder for innlie documentation but you can live without it. lib_Context-Scorer code is documenting itself.

* [Application Flow and Server Notes](https://github.com/Vocinity/context-scorer/blob/stable/doc/Application-Flow-Server-Notes.MD)
* [Usage Instructions and Client Notes](https://github.com/Vocinity/context-scorer/blob/stable/doc/Usage-Instructions-Client-Notes.MD)
* Training (in progress...)

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

#### lib_Context-Scorer
[From aMisc Inclusion](https://github.com/Vocinity/aMisc)
- External Deps

| **What**              | **Critical?**               | **Works?**   |**Best**    |**Notes**    |
|:---------------------:|:---------------------------:|:------------:|:----------:|:-----------:|
| Libtorch              | :heavy_check_mark yes       | >=1.8.2      | 1.10       |             |
| ONNX Runtime          | :heavy_check_mark yes       | >=1.7.0      | 1.10       | API changes at 1.10 but they are capable |
| CUDA SDK              | :white_check_mark nice      | >=11.0.3     | 11.4       | [ONNX Runtime](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements) defines compatibility            |
| CUDNN                 | :white_check_mark nice      | >=8.0.4      | 11.4       | [ONNX Runtime](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements) defines compatibility            |
| TensorRT              | :white_check_mark nice      | >=7.2        | 8.0        | [ONNX Runtime](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#requirements) defines compatibility            |

- Submodules
  - MAGIC_ENUM *is a must.*
  - RANGE_V3 *is optional for pre-C++20. (If you suffer from [compiler seg fault](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=96720), disable RANGE_V3 and use C++20)*
  - JSON *is a must.*
  - PYSTRING or RANGE_V3 or c++20 support *is needed.*
  - LEVENSHTEIN_SSE or RAPIDFUZZ_CPP *or both would be nice.*
  - SOUNDEX *is optional.*
  - DOUBLE_METAPHONE *is optional.*

#### Context-Scorer_Example
- lib_Context-Scorer 

#### Context-Scorer_Server
  - lib_Context-Scorer 
  - GRPC >= 1.25. v1.25 of RHEL8 and [1.27 deb](https://github.com/Vocinity/apt-rd#grpc-127-dev)
(Ubuntu Bionic & Focal versions are buggy) are used for development.
  - Protobuf which is compatible with your GRPC. RHEL8 rpm and [3.18 deb](https://github.com/Vocinity/apt-rd#protobuf-suite-dev)
 are used for development.

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

> Common  and lib_Context-Scorer

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
| ```USE_TORCH_CUDA_RT```     | ```0```            | This option makes use of libcudart in libtorch library path instead of the one that comes with CUDA SDK. Normally they can live together but some systems require to enable this switch.
| ```CUDNN_OFF```             | ```0```            | otherwise CUDNN in /usr/local/cuda/lib64 wont be be linked. Requires CUDA.
| ```TENSOR_RT_OFF```         | ```0```            | otherwise Tensor RT in /usr/local/trt/lib wont be be linked. Requires CUDA.
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

> grpc-server

| **Switch**                  | **Default Value**  |**By Default**                              |
|:---------------------------:|:------------------:|:-------------------------------------:|
| ```SERVER_OFF```            | ```0```            | Building server


#### Installation Material

> Context-Scorer

*One* of below depending relese|debug build and hw acceleration backend.
```
DEPL_ROOT/bin/akil/Context-Scorer_cu
DEPL_ROOT/bin/akil/Context-Scorer_cl
DEPL_ROOT/bin/akil/Context-Scorer_cpu
DEPL_ROOT/bin/akil/Context-Scorer_cu+dbg
DEPL_ROOT/bin/akil/Context-Scorer_cl+dbg
DEPL_ROOT/bin/akil/Context-Scorer_cpu+dbg
```

> Context-Scorer_Server

*One* of below depending relese|debug build and hw acceleration backend.
```
DEPL_ROOT/bin/akil/Context-Scorer_Server_cu
DEPL_ROOT/bin/akil/Context-Scorer_Server_cl
DEPL_ROOT/bin/akil/Context-Scorer_Server_cpu
DEPL_ROOT/bin/akil/Context-Scorer_Server_cu+dbg
DEPL_ROOT/bin/akil/Context-Scorer_Server_cl+dbg
DEPL_ROOT/bin/akil/Context-Scorer_Server_cpu+dbg
```

> lib_Context-Scorer

Release build of the commit 6691d7dcfeb5076db749a0cba25b48cfe5395379@branch and cpu backend would install:
```
DEPL_ROOT/lib/akil/lib_Context-Scorer_cpu.so
DEPL_ROOT/lib/akil/lib_Context-Scorer_cpu.so.2021-08-05_19:02@6691d7dcfeb5076db749a0cba25b48cfe5395379
DEPL_ROOT/lib/akil/lib_Context-Scorer_cpu.so.2021-08-05_19:02@6691d7dcfeb5076db749a0cba25b48cfe5395379.0
DEPL_ROOT/lib/akil/lib_Context-Scorer_cpu.so.2021-08-05_19:02@6691d7dcfeb5076db749a0cba25b48cfe5395379.0.0

DEPL_ROOT/include/akil/Context_Scorer.hpp

DEPL_ROOT/share/akil/qmake/depend_context-scorer.pri
```
For qmake build system, including only `depend_context-scorer.pri`is enough for everything; header, libraries, lower level dependencies...
Just like how Context_Scorer do.

> Conan

If in use, Conan dependencies and `conanbuildinfo.pri`go to `DEPL_ROOT/conan/pro-file-name-of-the-subproject`:
```
DEPL_ROOT/conan/context-scorer_example
DEPL_ROOT/conan/context-scorer_server
DEPL_ROOT/conan/context-scorer_library
```
and you will find `DEPL_ROOT/share/akil/qmake/pro-file-name-of-the-subproject_conan-linker-runtime.sh`:
```
DEPL_ROOT/share/akil/qmake/context-scorer-example_conan-linker-runtime.sh
DEPL_ROOT/share/akil/qmake/context-scorer-library_conan-linker-runtime.sh
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

#### Build Procedure

> Lets say we are on Centos, you know CENTOS is not fully supported, default gcc is 8.4, we have a nvidia gpu, our cpu has not igpu.

* Get some dependencies from yum
```bash
sudo yum install python3-devel qt5-qtbase-devel tbb-devel
```

* Put ONNX, CUDA SDK, CUDNN, TensorRT and Torch to (`CENTOS` or `else`) places under DEPS_ROOT (`/opt/local` is default) and `/usr/local` 
(for NVIDIA toolkits) where [aMisc](https://github.com/Vocinity/aMisc/blob/stable/qmake/depend_aMisc_template.pri)
expects to find.
* and make sure linker is able to see these libraries. Either by `export`ing `LD_LIBRARY_PATH` or ld.so.conf:
```
$ cat /etc/ld.so.conf.d/opt.conf 
/opt/local/lib/akil
/opt/local/lib/onnx
/opt/local/lib/torch
/usr/local/cuda/lib64
/usr/local/cudnn/lib64
/usr/local/trt/lib
```
(Compilation is locating libraries by known exact paths, also you should tell linker where can be your libSomething.so in runtime.)
* Prepare aMisc:
```bash
cd /home/vocinity/work
git clone --recurse-submodules https://github.com/Vocinity/aMisc
cd aMisc
mkdir build
cd build
```
* Setup gcc 10+ by sourcing secondary environment:
```bash
source /opt/rh/gcc-toolset-11/enable
```

- Configure aMisc:
  - OCV_OFF because we dont need it for Context-Scorer and it is an external dependency that requires you to compile yourself.
  - CL_OFF because we dont need OpenCL availability in CUDA build.
  - NO_CONAN because we are using yum.
  - QT_OFF so we just need qmake, we do not need Qt framework libraries for Context Scorer.
  - CENTOS because of Centos.
  - WT_OFF
```bash
qmake-qt5 ..  CONFIG+=OCV_OFF  CONFIG+=CL_OFF CONFIG+=NO_CONAN CONFIG+=QT_OFF CONFIG+=WT_OFF CONFIG+=CENTOS
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
  - CL_OFF because we dont need OpenCL availability in CUDA build.
  - NO_CONAN because we are using yum.
  - CENTOS because of Centos.

 ```bash
qmake-qt5 .. CONFIG+=CL_OFF CONFIG+=NO_CONAN CONFIG+=CENTOS
```
* Build Context Scorer:
```bash
make -j 8
make install
```

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
### - January 27th -
* Server Instructions README

### - January 20th -
* Client Instructions README

### - January 18th -
* FP16 Support

### - October 30th -
* Release
