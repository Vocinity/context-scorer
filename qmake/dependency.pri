CUDA_AVAILABLE{include(cuda.pri)}
#----------------------------------------------------
include($${DEPS_ROOT}/share/akil/qmake/dependMisc.pri)
#----------------------------------------------------

SRC_DIR=$$PWD/../src/
INCLUDEPATH += $${SRC_DIR}
HEADERS += $${SRC_DIR}/Context-Scorer.hpp \
           $${SRC_DIR}/backends/Faster_Transformer-Scorer-Backend.hpp \
           $${SRC_DIR}/backends/Abstract-Scorer-Backend.hpp \
           $${SRC_DIR}/backends/Torch-Scorer-Backend.hpp \
           $${SRC_DIR}/backends/LightSeq-Scorer-Backend.hpp \
           $${SRC_DIR}/Homophonic-Alternatives.hpp

SOURCES += $${SRC_DIR}/Context-Scorer.cpp \
           $${SRC_DIR}/Tokenizer.cpp \
           $${SRC_DIR}/Homophonic-Alternatives.cpp

LIGHTSEQ_AVAILABLE{
    PROTOS+=$$PWD/../3rdparty/lightseq/lightseq/inference/proto/gpt.proto
    PROTOPATH =
    include(protobuf.pri)
}

THIRD_PARTY_SRC=$$PWD/../3rdparty/
CUDA_AVAILABLE{
    LIGHTSEQ_AVAILABLE{
        CENTOS{
        #
        }else{
            LIBS+= -L/usr/lib/x86_64-linux-gnu/hdf5/serial/ -lhdf5
        }
        INCLUDEPATH += $${THIRD_PARTY_SRC}/lightseq/lightseq/inference
        SOURCES+= $${THIRD_PARTY_SRC}/lightseq/lightseq/inference/proto/gpt_weight.cc
    }
    FASTER_TRANSFORMER_AVAILABLE{
        INCLUDEPATH += $${THIRD_PARTY_SRC}/FasterTransformer/
        SOURCES+= $${THIRD_PARTY_SRC}/FasterTransformer/src/fastertransformer/models/gpt/Gpt.cc \
                  $${THIRD_PARTY_SRC}/FasterTransformer/src/fastertransformer/models/gpt/GptContextDecoder.cc \
                  $${THIRD_PARTY_SRC}/FasterTransformer/src/fastertransformer/models/gpt/GptDecoder.cc \
                  $${THIRD_PARTY_SRC}/FasterTransformer/src/fastertransformer/models/gpt/GptWeight.cc \
                  $${THIRD_PARTY_SRC}/FasterTransformer/src/fastertransformer/models/gpt/GptDecoderLayerWeight.cc \
                  $${THIRD_PARTY_SRC}/FasterTransformer/src/fastertransformer/utils/cublasMMWrapper.cc \
                  #$${THIRD_PARTY_SRC}/FasterTransformer/src/fastertransformer/utils/cublasINT8MMWrapper.cc \
                  $${THIRD_PARTY_SRC}/FasterTransformer/src/fastertransformer/utils/cublasAlgoMap.cc \
                  $${THIRD_PARTY_SRC}/FasterTransformer/src/fastertransformer/layers/FfnLayer.cc \
                  $${THIRD_PARTY_SRC}/FasterTransformer/src/fastertransformer/layers/attention_layers/GptContextAttentionLayer.cc

    }
}

DISTFILES+= $${SRC_DIR}/../scorer.py
