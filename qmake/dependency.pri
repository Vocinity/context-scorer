include($${DEPS_ROOT}/share/akil/qmake/dependMisc.pri)

#----------------------------------------------------

SRC_DIR=$$PWD/../src/
INCLUDEPATH += $${SRC_DIR}
HEADERS += $${SRC_DIR}/Context_Scorer.hpp \
           $${SRC_DIR}/Homophonic_Alternatives.hpp

SOURCES += $${SRC_DIR}/Context_Scorer.cpp \
           $${SRC_DIR}/Tokenizer.cpp \
           $${SRC_DIR}/Homophonic_Alternatives.cpp

THIRD_PARTY_SRC=$$PWD/../3rdparty/
CUDA_AVAILABLE{
    LIGHTSEQ_AVAILABLE{
        INCLUDEPATH+= /usr/include/hdf5/serial/
        INCLUDEPATH += $${THIRD_PARTY_SRC}/lightseq/lightseq/inference
        SOURCES+= $${THIRD_PARTY_SRC}/lightseq/lightseq/inference/proto/gpt_weight.cc
    }
}

DISTFILES+= $${SRC_DIR}/../scorer.py
