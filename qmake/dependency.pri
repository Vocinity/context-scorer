CUDA_AVAILABLE{include(cuda.pri)}
#----------------------------------------------------
include($${DEPS_ROOT}/share/akil/qmake/depend_aMisc.pri)
#----------------------------------------------------

SRC_DIR=$$PWD/../src/
INCLUDEPATH += $${SRC_DIR}
HEADERS += $${SRC_DIR}/Context-Scorer.hpp \
           $${SRC_DIR}/backends/Abstract-Scorer-Backend.hpp \
           $${SRC_DIR}/backends/Torch-Scorer-Backend.hpp \
           $${SRC_DIR}/backends/ONNX-Scorer-Backend.hpp \
           $${SRC_DIR}/Homophonic-Alternatives.hpp

SOURCES += $${SRC_DIR}/Context-Scorer.cpp \
           $${SRC_DIR}/Tokenizer.cpp \
           $${SRC_DIR}/Homophonic-Alternatives.cpp

DISTFILES+= $${SRC_DIR}/../scorer.py
