message($$TARGET)

DISTFILES+=README.md

include(qmake/subdirVariablePassing.pri)

TEMPLATE= subdirs
CONFIG += ordered

SUBDIRS+=library
library.file=qmake/context-scorer_library.pro

!EXAMPLE_OFF{
    message("Enabled building example program")
    SUBDIRS+=example
    example.file=example/context-scorer_example.pro
    example.depends+=library
}

!SERVER_OFF{
    message("Enabled building gRPC server")
    SUBDIRS+=grpc-server
    grpc-server.file=grpc-server/context-scorer_server.pro
    grpc-server.depends+=library
}
