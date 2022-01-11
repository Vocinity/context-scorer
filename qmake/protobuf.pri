unix {
    isEmpty(PROTOS){
        message("You should list PROTOS before including protobuf.pri, nothing listed so not running protoc.")
    }
    else{
        GRPC{
            CENTOS{
                message("GRPC protobuf extension enabled. Expected to be found in path: "$$system(which grpc_cpp_plugin))
                LIBS+=-lgrpc++
            }
            else{
            message("GRPC protobuf extension enabled. Expected to be found in path: "$${DEPS_ROOT}/bin/grpc_cpp_plugin)
            LIBS+= -L$${DEPS_ROOT}/lib/grpc -l:libgrpc++.so}
        }
        CENTOS{LIBS+= -L/usr/lib64/ -lprotobuf -lprotoc}
        else{
            LIBS+= -L$${DEPS_ROOT}/lib -l:libprotobuf.so.3.18.0.0 -l:libprotoc.so.3.18.0.0
        }

        CONFIG+=PROTO_PROCESSING
        CENTOS{isEmpty(PROTOC):PROTOC = $$system(which protoc)}
        else{isEmpty(PROTOC):PROTOC = $${DEPS_ROOT}/bin/protoc}
        message("protoc $$PROTOC will run for $${PROTOS}")

        for(p1, PROTOS):PROTOPATH += $$clean_path($$dirname(p1))
        for(p2, PROTOPATH):PROTOPATHS += --proto_path=$${p2}

        protobuf_decl.name = protobuf headers
        protobuf_decl.input = PROTOS
        protobuf_decl.output = ${QMAKE_FILE_IN_PATH}/${QMAKE_FILE_BASE}.pb.h
        GRPC{
            CENTOS{
                protobuf_decl.commands = $$PROTOC --cpp_out=${QMAKE_FILE_IN_PATH} $${PROTOPATHS} ${QMAKE_FILE_BASE}.proto\
                --grpc_out=${QMAKE_FILE_IN_PATH} --plugin=protoc-gen-grpc=$$system(which grpc_cpp_plugin)
            }
            else{
                protobuf_decl.commands = $$PROTOC --cpp_out=${QMAKE_FILE_IN_PATH} $${PROTOPATHS} ${QMAKE_FILE_BASE}.proto\
                --grpc_out=${QMAKE_FILE_IN_PATH} --plugin=protoc-gen-grpc=$${DEPS_ROOT}/bin/grpc_cpp_plugin
            }
        }else{
            protobuf_decl.commands = $$PROTOC --cpp_out=${QMAKE_FILE_IN_PATH} $${PROTOPATHS} ${QMAKE_FILE_BASE}.proto
        }
        protobuf_decl.variable_out = HEADERS
        #protobuf_decl.CONFIG = target_predeps
        QMAKE_EXTRA_COMPILERS += protobuf_decl
        #INCLUDEPATH+=$${OUT_PWD}/

        protobuf_impl.name = protobuf sources
        protobuf_impl.input = PROTOS
        protobuf_impl.output = ${QMAKE_FILE_IN_PATH}/${QMAKE_FILE_BASE}.pb.cc
        protobuf_impl.depends =  ${QMAKE_FILE_IN_PATH}/${QMAKE_FILE_BASE}.pb.h
        protobuf_impl.commands = $$escape_expand(\n\n)
        protobuf_impl.variable_out = SOURCES
        #protobuf_impl.CONFIG = target_predeps
        QMAKE_EXTRA_COMPILERS += protobuf_impl

        GRPC{
            protobuf_grpc_decl.name = protobuf grpc headers
            protobuf_grpc_decl.input = PROTOS
            protobuf_grpc_decl.output = ${QMAKE_FILE_IN_PATH}/${QMAKE_FILE_BASE}.grpc.pb.h
            protobuf_grpc_decl.depends = ${QMAKE_FILE_IN_PATH}/${QMAKE_FILE_BASE}.pb.cc
            protobuf_grpc_decl.commands = $$escape_expand(\n\n)
            protobuf_grpc_decl.variable_out = HEADERS
            #protobuf_grpc_decl.CONFIG = target_predeps
            QMAKE_EXTRA_COMPILERS += protobuf_grpc_decl

            protobuf_grpc_impl.name = protobuf grpc sources
            protobuf_grpc_impl.input = PROTOS
            protobuf_grpc_impl.output =  ${QMAKE_FILE_IN_PATH}/${QMAKE_FILE_BASE}.grpc.pb.cc
            protobuf_grpc_impl.depends =  ${QMAKE_FILE_IN_PATH}/${QMAKE_FILE_BASE}.grpc.pb.h
            protobuf_grpc_impl.commands = $$escape_expand(\n\n)
            protobuf_grpc_impl.variable_out = SOURCES
            #protobuf_grpc_impl.CONFIG = target_predeps
            QMAKE_EXTRA_COMPILERS += protobuf_grpc_impl
        }
    }
}
