CUDA_AVAILABLE{include(cuda.pri)}
#----------------------------------------------------
include($${DEPS_ROOT}/share/akil/qmake/dependMisc.pri)
#----------------------------------------------------

SRC_DIR=$$PWD/../src/
INCLUDEPATH += $${SRC_DIR}
HEADERS += $${SRC_DIR}/Context_Scorer.hpp \
           $${SRC_DIR}/Homophonic_Alternatives.hpp

SOURCES += $${SRC_DIR}/Context_Scorer.cpp \
           $${SRC_DIR}/Tokenizer.cpp \
           $${SRC_DIR}/Homophonic_Alternatives.cpp

PROTOS+=$$PWD/../3rdparty/lightseq/lightseq/inference/proto/gpt.proto
PROTOPATH =
include(protobuf.pri)

THIRD_PARTY_SRC=$$PWD/../3rdparty/
CUDA_AVAILABLE{
    LIGHTSEQ_AVAILABLE{
        CENTOS{
        }else{
            LIBS+= -L/usr/lib/x86_64-linux-gnu/hdf5/serial/ -lhdf5
        }
        INCLUDEPATH += $${THIRD_PARTY_SRC}/lightseq/lightseq/inference
        SOURCES+= $${THIRD_PARTY_SRC}/lightseq/lightseq/inference/proto/gpt_weight.cc
    }
}

DISTFILES+= $${SRC_DIR}/../scorer.py
