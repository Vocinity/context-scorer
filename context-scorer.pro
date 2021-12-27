message($$TARGET)

DISTFILES+=README.md

include(qmake/subdirVariablePassing.pri)

TEMPLATE= subdirs
CONFIG += ordered

SUBDIRS+=library
library.file=qmake/context_scorer-library.pro

#!TEST_OFF{
#    message("Compiling tests")
#    SUBDIRS+=test
#    test.depends=qmake/noiseReductionLibrary.pro
#}

!EXAMPLE_OFF{
    message("Enabled example program")
    SUBDIRS+=example
    example.file=example/context_scorer-example.pro
    example.depends+=library
}

#!GST_PLUGIN_OFF{
#    message("Enabled gst plugin")
#    SUBDIRS+=gst
#    gst.file=gst/gst-scorer.pro
#    gst.depends+=library
#}
