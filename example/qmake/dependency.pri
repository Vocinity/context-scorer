unix {
    include($${DEPS_ROOT}/share/akil/qmake/dependMisc.pri)

    MY_BUILD_PATH=$${OUT_PWD}/../qmake/bin/
    CONFIG(release, debug|release) {
        CUDA_AVAILABLE{LIBS += -L$${MY_BUILD_PATH}/ -l_Context-Scorer_cu}
        else{
            CL_AVAILABLE{
                LIBS += -L$${MY_BUILD_PATH}/ -l_Context-Scorer_cl
            }else{
                LIBS += -L$${MY_BUILD_PATH}/ -l_Context-Scorer_cpu
            }
        }
    }

    CONFIG(debug, debug|release) {
        CUDA_AVAILABLE{LIBS += -L$${MY_BUILD_PATH} -l_Context-Scorer_cu+dbg}
        else{
            CL_AVAILABLE{
                LIBS += -L$${MY_BUILD_PATH}/ -l_Context-Scorer_cl+dbg
            }else{
                LIBS += -L$${MY_BUILD_PATH}/ -l_Context-Scorer_cpu+dbg
                }
            }
    }

    !ON_CONAN{
        LIGHTSEQ_AVAILABLE{
            CENTOS{
            }else{
                LIBS+= -L/usr/lib/x86_64-linux-gnu/hdf5/serial/ -lhdf5
            }
            LIBS+= -L$${DEPS_ROOT}/lib/ -l:libprotobuf.so.30.0.1 -l:libprotobuf-lite.so.30.0.1 -l:libprotoc.so.30.0.1
        }
    }else{
        CONFIG += conan_basic_setup
        Context_Scorer_Conan_Deploy-Dir=$${DEPL_ROOT}/conan/
        Context_Scorer_Conan_Pri=$${Context_Scorer_Conan_Deploy_Dir}/context_scorer_library/conanbuildinfo.pri
        include($${Context_Scorer_Conan_Pri})
    }
}

#----------------------------------------------------
SRC_DIR=$$PWD/../src/
INCLUDEPATH += $${SRC_DIR}
SOURCES += $${SRC_DIR}/main.cpp
