unix {
    isEmpty(PROTOS){
        message("You should list PROTOS before including protobuf.pri, nothing listed so not running protoc.")
    }
    else{
        CONFIG+=PROTO_PROCESSING
        isEmpty(PROTOC):PROTOC = $${DEPS_ROOT}/bin/protoc
        message("$$PROTOC is running for $${PROTOS}")

        LIBS+= -L$${DEPS_ROOT}/lib/ -l:libprotobuf.so.30.0.1 -l:libprotobuf-lite.so.30.0.1 -l:libprotoc.so.30.0.1

        for(p1, PROTOS):PROTOPATH += $$clean_path($$dirname(p1))
        for(p2, PROTOPATH):PROTOPATHS += --proto_path=$${p2}

        protobuf_decl.name = protobuf headers
        protobuf_decl.input = PROTOS
        protobuf_decl.output = ${QMAKE_FILE_BASE}.pb.h
        protobuf_decl.commands = $$PROTOC --cpp_out="." $${PROTOPATHS} ${QMAKE_FILE_BASE}.proto
        protobuf_decl.variable_out = HEADERS
        QMAKE_EXTRA_COMPILERS += protobuf_decl
        INCLUDEPATH+=$${OUT_PWD}/

        protobuf_impl.name = protobuf sources
        protobuf_impl.input = PROTOS
        protobuf_impl.output = ${QMAKE_FILE_BASE}.pb.cc
        protobuf_impl.depends = ${QMAKE_FILE_BASE}.pb.h
        protobuf_impl.commands = $$escape_expand(\n)
        protobuf_impl.variable_out = SOURCES
        QMAKE_EXTRA_COMPILERS += protobuf_impl
    }
}
