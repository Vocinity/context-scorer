exists("$${PWD}/cross.pri") {

    message("Using cross compilation config file $${PWD}/cross.pri. If you want a native compilation, simply rename that file to something else.")
    include($${PWD}/cross.pri)

    QMAKE_CFLAGS+=$$ENABLED_INSTRUCTIONS $$join(CPU_ARCH, "",'-march=', '')
    QMAKE_CXXFLAGS+=$$ENABLED_INSTRUCTIONS $$join(CPU_ARCH, "",'-march=', '')
    DEFINES+=PRESETS
    CONFIG+=PRESETS

        !NO_AVX_PLEASE{
            message("AVX is enabled.")
            QMAKE_CXXFLAGS += -mavx
            QMAKE_CFLAGS += -mavx
            CONFIG+= AVX
            DEFINES+=AVX
        }
        !NO_AVX2_PLEASE{
            message("AVX2 is enabled.")
            QMAKE_CXXFLAGS += -mavx2
            QMAKE_CFLAGS += -mavx2
        }
        !NO_BMI_PLEASE{
            message("BMI and BMI2 are enabled.")
            QMAKE_CXXFLAGS += -mbmi -mbmi2
            QMAKE_CFLAGS += -mbmi -mbmi2
        }
        CONFIG+=SSE41
        DEFINES+=SSE41
}
else{
        ENABLED_CONSTANT_USED_IN_TEXT_WHICH_RETURNS_FROM_SYSTEM_CPU_QUERY="[enabled]"
        INSTRUCTIONS_TO_QUERY=msse msse2 msse3 mssse3 msse4.1 msse4.2 mavx mavx2 mfma mbmi mbmi2

        !CELERON{
                !ARM_DEVICE{
                        !INTEL_I_SERIES__RYZEN{
                                CONFIG+=AUTO_DESKTOP
                        }
                }
        }
        CELERON{
                         message("Basic desktop instructions are enabled by CELERON flag. This configuration is for older desktop level CPUs. No AVX and AVX2.")
                         QMAKE_CXXFLAGS += -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2
                         QMAKE_CFLAGS += -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2
                         DEFINES+=NO_AVX_PLEASE NO_AVX2_PLEASE NO_BMI_PLEASE SSE41
                         CONFIG+=NO_AVX_PLEASE NO_AVX2_PLEASE NO_BMI_PLEASE SSE41
        }
        ARM_DEVICE{
                         message("ARM instructions are enabled by ARM flag. This configuration is for embedded SOCs.")
                         QMAKE_CXXFLAGS +=
                         QMAKE_CFLAGS +=
                         DEFINES+=NO_AVX_PLEASE NO_AVX2_PLEASE NO_BMI_PLEASE RPI
                         CONFIG+=NO_AVX_PLEASE NO_AVX2_PLEASE NO_BMI_PLEASE RPI
        }
        XEON{
                        message("XEON instructions are enabled by XEON flag. This configuration is for AVX2less cpu's.")
                        QMAKE_CXXFLAGS += -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mavx -mfma
                        QMAKE_CFLAGS += -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mavx -mfma
                        DEFINES+=NO_AVX2_PLEASE NO_BMI_PLEASE SSE41 AVX
                        CONFIG+=NO_AVX2_PLEASE NO_BMI_PLEASE SSE41 AVX
        }
        INTEL_I_SERIES__RYZEN{
                        message("Up-to-date desktop instructions are enabled by INTEL_I_SERIES__RYZEN flag. This configuration is for Intel I3, I5, I7, I9 Series and RYZEN Arch")
                        QMAKE_CXXFLAGS += -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mavx -mavx2 -mfma -mbmi -mbmi2
                        QMAKE_CFLAGS += -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mavx -mavx2 -mfma -mbmi -mbmi2
                        DEFINES+=SSE41 AVX
                        CONFIG+=SSE41 AVX
        }

        AUTO_DESKTOP{
                        message("Using native CPU instructions.")
                        for(AN_INSTRUCTION, INSTRUCTIONS_TO_QUERY){
                                AN_INSTRUCTION_IN_GOOD_FORMAT=$$join(AN_INSTRUCTION, "",'"', ' "')
                                AVAILABILITY=$$system(gcc -march=native -Q --help=target|grep $$AN_INSTRUCTION_IN_GOOD_FORMAT)
                                contains(AVAILABILITY,$$ENABLED_CONSTANT_USED_IN_TEXT_WHICH_RETURNS_FROM_SYSTEM_CPU_QUERY){
                                        message($$AVAILABILITY)
                                        QMAKE_CFLAGS +=$$join(AN_INSTRUCTION, "",'-', '')
                                        QMAKE_CXXFLAGS +=$$join(AN_INSTRUCTION, "",'-', '')
                                    equals(AN_INSTRUCTION,"msse4.1"){
                                        DEFINES+=SSE41
                                        CONFIG+=SSE41
                                    }
                                    equals(AN_INSTRUCTION,"mavx"){
                                        DEFINES+=AVX
                                        CONFIG+=AVX
                                    }
                                }
                                else{
                                    equals(AN_INSTRUCTION,"mbmi"){
                                        DEFINES+=NO_BMI_PLEASE
                                        CONFIG+=NO_BMI_PLEASE
                                    }
                                    equals(AN_INSTRUCTION,"mbmi2"){
                                        DEFINES+=NO_BMI_PLEASE
                                        CONFIG+=NO_BMI_PLEASE
                                    }
                                    equals(AN_INSTRUCTION,"mavx"){
                                        DEFINES+=NO_AVX_PLEASE
                                        CONFIG+=NO_AVX_PLEASE
                                    }
                                    equals(AN_INSTRUCTION,"mavx2"){
                                        DEFINES+=NO_AVX2_PLEASE
                                        CONFIG+=NO_AVX2_PLEASE
                                    }
                                }
                        }
                        CPU_GEN_QUERY=$$system("gcc -march=native -Q --help=target|grep march | head -n 1")
                        CPU_GEN=$$replace(CPU_GEN_QUERY, "=", "")
                        CPU_GEN_IN_GOOD_FORMAT=$$last(CPU_GEN)
                        message("Compiling for" $$CPU_GEN_IN_GOOD_FORMAT)
                        QMAKE_CFLAGS += $$join(CPU_GEN_IN_GOOD_FORMAT, "",'-march=', '')
                        QMAKE_CXXFLAGS += $$join(CPU_GEN_IN_GOOD_FORMAT, "",'-march=', '')
        }
}
