unix {
    #
    # Qt qmake integration with Google Protocol Buffers compiler protoc
    #
    # To compile protocol buffers with qt qmake, specify PROTOS variable and
    # include this file
    #
    # Example:
    # LIBS += /usr/local/lib/libprotobuf.so
    # PROTOS = a.proto b.proto
    # include(protobuf.pri)
    #
    # By default protoc looks for .proto files (including the imported ones) in
    # the current directory where protoc is run. If you need to include additional
    # paths specify the PROTOPATH variable
    #

    isEmpty(PROTOS):error("Define PROTOS before including protobuf.pri")
    message("Protobuf processor is running for $${PROTOS}")
    isEmpty(PROTOC):PROTOC = protoc

    LIBS+= -L$${DEPS_ROOT}/lib/akil -lprotobuf

    for(p1, PROTOS):PROTOPATH += $$clean_path($$dirname(p1))
    for(p2, PROTOPATH):PROTOPATHS += --proto_path=$${p2}

    protobuf_decl.name = protobuf header
    protobuf_decl.input = PROTOS
    protobuf_decl.output = ${QMAKE_FILE_BASE}.pb.h
    protobuf_decl.commands = protoc --cpp_out="." $${PROTOPATHS} ${QMAKE_FILE_BASE}.proto
    protobuf_decl.variable_out = GENERATED_FILES
    QMAKE_EXTRA_COMPILERS += protobuf_decl

    PROTOBUF_HEADERS =
    for(proto, PROTOS) {
            headers = $$replace(proto, .proto, .pb.h)
            #message("headers: $${headers}")
            for (header, headers) {
                    PROTOBUF_HEADERS += $${header}
                    #message("header: $${header}")
            }
    }
    HEADERS += $${PROTOBUF_HEADERS}

    protobuf_impl.name = protobuf implementation
    protobuf_impl.input = PROTOS
    protobuf_impl.output = ${QMAKE_FILE_BASE}.pb.cc
    protobuf_impl.depends = ${QMAKE_FILE_BASE}.pb.h $${PROTOBUF_HEADERS}
    protobuf_impl.commands = $$escape_expand(\n)
    protobuf_impl.variable_out = GENERATED_SOURCES
    QMAKE_EXTRA_COMPILERS += protobuf_impl
}
