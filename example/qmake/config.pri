TEMPLATE = app
CONFIG+= console
CONFIG+= no_keywords
VER_DATE=$$system(git log -1 --date=format:"%Y-%m-%d_%H:%M" --format="%ad")
VER_HASH = $$system(git describe --always --abbrev=0)
VERSION=$${VER_DATE}@$${VER_HASH}

unix{
    message("-------------------Example program-------------------")
    message("Active branch is: "$$system('git symbolic-ref --short HEAD'))

    DEPS_ROOT_ENV_VAR=$$(DEPS_ROOT)
    isEmpty(DEPS_ROOT): DEPS_ROOT=$$DEPS_ROOT_ENV_VAR
    if(isEmpty(DEPS_ROOT) | equals(DEPS_ROOT, 1)){
        DEPS_ROOT=/opt/local
    }

    DEPL_ROOT_ENV_VAR=$$(DEPL_ROOT)
    isEmpty(DEPL_ROOT): DEPL_ROOT=$$DEPL_ROOT_ENV_VAR
    if(isEmpty(DEPL_ROOT) | equals(DEPL_ROOT, 1)){
        DEPL_ROOT=/opt/local
    }
    message("Installing into the $$DEPL_ROOT")
    BUILD_ROOT=$$PKG_BUILD_ROOT
    isEmpty(BUILD_ROOT):BUILD_ROOT=$$(PKG_BUILD_ROOT)
    !isEmpty(BUILD_ROOT){
        message("Build root is $${BUILD_ROOT}")
        DEPL_ROOT=$${BUILD_ROOT}/$${DEPL_ROOT}
    }
    linux-rasp-pi4-v3d-g++{
        message("ARM Build")
        DEPS_ROOT=$$[QT_SYSROOT]/$$DEPS_ROOT
        DEPL_ROOT=$$[QT_SYSROOT]/$$DEPL_ROOT
    }
    linux-g++{
        message("Desktop Build")
    }

    CONFIG(release, debug|release) {
        message("Release Mode")
    }

    CONFIG(debug, debug|release) {
        message("Debug Mode")
    }

    GCC8{
        QMAKE_CC = gcc-8
        QMAKE_CXX = g++-8
    }
    GCC9{
        QMAKE_CC = gcc-9
        QMAKE_CXX = g++-9
    }
    GCC10{
        QMAKE_CC = gcc-10
        QMAKE_CXX = g++-10
    }
    GCC11{
        QMAKE_CC = gcc-11
        QMAKE_CXX = g++-11
    }

    !GCC_CHECK_OFF{
        CompilerFullVersionToBeParsed=$$system($$QMAKE_CXX " -dumpfullversion")
        CompilerFullVersionSplitted=$$split(CompilerFullVersionToBeParsed,.)
        CompilerMajorVersion=$$first(CompilerFullVersionSplitted)
        CompilerMinorVersion=$$member(CompilerFullVersionSplitted,1,1)
        CompilerPatchVersion=$$last(CompilerFullVersionSplitted)

        message("GCC $${CompilerMajorVersion}.$${CompilerMinorVersion}.$${CompilerPatchVersion} in use")

        greaterThan(CompilerMajorVersion, 7){
            isEqual(CompilerMajorVersion,8){
                greaterThan(CompilerMinorVersion, 2):CONFIG+=CPP17_AVAILABLE
            }else:CONFIG+=CPP17_AVAILABLE
        }

        greaterThan(CompilerMajorVersion, 9){
            isEqual(CompilerMajorVersion,10){
                greaterThan(CompilerMinorVersion, 0):CONFIG+=CPP20_AVAILABLE
            }else:CONFIG+=CPP20_AVAILABLE
        }

        !CPP17:!CPP20_AVAILABLE{
            message("GCC $${CompilerMajorVersion}.$${CompilerMinorVersion} is not sufficient for C++20")
            message("For C++20 support, you can set GCC10 or GCC11 CONFIG switches after installing them.")
            CPP17_AVAILABLE{
                message("CPP17 support is available, switching to gnu++17")
                CONFIG+=CPP17
            }else{
                error("GCC $${CompilerMajorVersion}.$${CompilerMinorVersion}.$${CompilerPatchVersion} is not sufficient for C++20 and even C++17")
            }
        }
    }

    CPP17{
        QMAKE_CXXFLAGS += -std=gnu++17
        DEFINES+=CPP17_AVAILABLE
        message("CPP 17")
    }else{
        QMAKE_CXXFLAGS += -std=gnu++20 -fcoroutines
        DEFINES+=CPP20_AVAILABLE
        message("CPP 20")
    }

    CONFIG+=-D_GLIBCXX_PARALLEL
    QMAKE_CFLAGS_DEBUG +=-O0
    QMAKE_CFLAGS_RELEASE +=-Ofast -ffast-math
    QMAKE_CFLAGS += -fopenmp -fopenmp-simd
    QMAKE_CFLAGS += -ffp-contract=fast
    #----------------------------------------
    QMAKE_CXXFLAGS_DEBUG +=-O0
    QMAKE_CXXFLAGS_RELEASE += -Ofast -ffast-math -funroll-all-loops -fpeel-loops\
    -ftracer -ftree-vectorize
    !LTO_OFF{
        message("LTO is on")
        QMAKE_CXXFLAGS_RELEASE +=-flto
    }
    QMAKE_CXXFLAGS_WARN_ON = -Wall -Wextra -Wno-unused-function -Wno-comment
    QMAKE_CXXFLAGS+= -fopenmp -fopenmp-simd

    !CCACHE_OFF{
        QMAKE_CXX = ccache $$QMAKE_CXX
        QMAKE_CC = ccache $$QMAKE_CC
    }

    linux-rasp-pi4-v3d-g++{
            CONFIG+=ARM_DEVICE
            DEFINES+=RPI_4
        }

    include(commonConfig.pri)

    defineTest(enableExtension) {
        extensionName=$$1
        message("$${extensionName} is enabled")
        DEFINES+=$${extensionName}_AVAILABLE
        CONFIG+=$${extensionName}_AVAILABLE
        export(DEFINES)
        export(CONFIG)
    }

    defineTest(extensionProcessor) {
        extensionName=$$1
        !$${extensionName}_OFF{
            enableExtension($$extensionName)
        }else{
            message("$${extensionName} is disabled")
        }
    }

    !CUDA_OFF{
        if(exists("/usr/local/cuda/version.txt")|exists("/usr/local/cuda/version.json")){
            message("FOUND CUDA")
            DEFINES+=CUDA_AVAILABLE
            CONFIG+=CUDA_AVAILABLE
            extensionProcessor(THRUST)
            include(cuda.pri)
         }
    }

    Release:include(conan.pri)
    Debug:include(conan.pri)

    extensionProcessor(CL)
    QT -= core gui qml quickcontrols2 quickcontrols qtquickcompiler
    CONFIG -= qt
    enableExtension(JSON)
    extensionProcessor(PYSTRING)
    extensionProcessor(SOUNDEX)
    extensionProcessor(DOUBLE_METAPHONE)
    extensionProcessor(LEVENSHTEIN_SSE)
    extensionProcessor(RAPIDFUZZ_CPP)
    enableExtension(ROBIN_HOOD_HASHING)
    enableExtension(MAGIC_ENUM)
    extensionProcessor(RANGE_V3)
    enableExtension(TORCH)

    #    CONFIG(debug, debug|release) {
    #    CONFIG += sanitizer
    #    CONFIG += sanitize_address
    # #   CONFIG += sanitize_memory
    #    CONFIG += sanitize_undefined
    # #   CONFIG += sanitize_thread
    #    }
}
