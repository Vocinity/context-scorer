include($${DEPS_ROOT}/share/akil/qmake/dependMisc.pri)

#----------------------------------------------------

SRC_DIR=$$PWD/../src/
INCLUDEPATH += $${SRC_DIR}
HEADERS += $${SRC_DIR}/Context_Scorer.hpp \
           $${SRC_DIR}/Homophonic_Alternatives.hpp

SOURCES += $${SRC_DIR}/Context_Scorer.cpp \
           $${SRC_DIR}/Tokenizer.cpp \
           $${SRC_DIR}/Homophonic_Alternatives.cpp


DISTFILES+= $${SRC_DIR}/../scorer.py
