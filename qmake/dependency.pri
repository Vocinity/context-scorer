include($${DEPS_ROOT}/share/akil/qmake/dependMisc.pri)

#----------------------------------------------------

SRC_DIR=$$PWD/../src/
INCLUDEPATH += $${SRC_DIR}
HEADERS += $${SRC_DIR}/Context_Scorer.hpp

SOURCES += $${SRC_DIR}/Context_Scorer.cpp \
           $${SRC_DIR}/Tokenizer.cpp

DISTFILES+= $${SRC_DIR}/../scorer.py
