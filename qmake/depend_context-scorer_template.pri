unix {
    MY_DEPS_ROOT=DEPS_DIR_HERE
    MY_DEPL_ROOT=DEPL_DIR_HERE
    include($${NR_DEPS_ROOT}/share/akil/qmake/dependMisc.pri)

    CONFIG(release, debug|release) {
        CUDA_AVAILABLE{LIBS += -L$${MY_DEPL_ROOT}/lib/akil -l:lib-Context-Scorer_cu.so}
        else{
            CL_AVAILABLE{
                LIBS += -L$${MY_DEPL_ROOT}/lib/akil -l:lib-Context-Scorer_cl.so
            }else{
                LIBS += -L$${MY_DEPL_ROOT}/lib/akil -l:lib-Context-Scorer_cpu.so
            }
        }
    }

    CONFIG(debug, debug|release) {
        CUDA_AVAILABLE{LIBS += -L$${MY_DEPL_ROOT}/lib/akil -l:lib-Context-Scorer_cu+dbg.so}
        else{
            CL_AVAILABLE{
                LIBS += -L$${MY_DEPL_ROOT}/lib/akil -l:lib-Context-Scorer_cl+dbg.so
            }else{
                LIBS += -L$${MY_DEPL_ROOT}/lib/akil -l:lib-Context-Scorer_cpu+dbg.so
                }
            }
    }

    !ON_CONAN{
    }else{
        CONFIG += conan_basic_setup
        Context_Scorer-Conan_Deploy_Dir=$${MY_DEPL_ROOT}/conan/
        Context_Scorer_Conan_Pri=$${Context_Scorer_Conan_Deploy_Dir}/context-scorer_library/conanbuildinfo.pri
        include($${Context_Scorer_Conan_Pri})
    }
}
