UI_DIR = bin/uics
MOC_DIR = bin/mocs
OBJECTS_DIR = bin/objs
UI_HEADERS_DIR=bin/ui
UI_SOURCES_DIR=bin/ui
RCC_DIR=bin/rcc
DESTDIR=bin

unix {
    CONFIG(release, debug|release) {
        CUDA_AVAILABLE{TARGET = Context-Scorer_cu}
        else{
            CL_AVAILABLE{
                TARGET = Context-Scorer_cl
            }else{
                TARGET = Context-Scorer_cpu
            }
        }
    }

    CONFIG(debug, debug|release) {
        CUDA_AVAILABLE{TARGET = Context-Scorer_cu+dbg}
        else{
            CL_AVAILABLE{
                TARGET = Context-Scorer_cl+dbg
            }else{
                TARGET = Context-Scorer_cpu+dbg
            }
        }
    }

    isEmpty(FAKE_INSTALL): FAKE_INSTALL=$$(FAKE_INSTALL)
    !isEmpty(FAKE_INSTALL){
        FORCELINK_CPP_FILE = force_link.cpp
        forcelink.target = $$FORCELINK_CPP_FILE
        forcelink.depends = FORCE
        forcelink.commands = touch $$FORCELINK_CPP_FILE
        QMAKE_EXTRA_TARGETS += forcelink
        PRE_TARGETDEPS+=$$forcelink.target
        SOURCES += $$FORCELINK_CPP_FILE
        !build_pass : write_file($$FORCELINK_CPP_FILE)

        isEqual(TEMPLATE,lib){
            targetPrefix=lib
            targetSuffix=.so
            targetName=$${targetPrefix}$${TARGET}$${targetSuffix}.$${VERSION}
            targetNames=$${targetName} $${targetName}.0 $${targetName}.0.0
            for(oneOfTargets, targetNames):{
                postCommandBody += $${OUT_PWD}/bin/$$oneOfTargets
            }
            postCommandBody+=$${OUT_PWD}/bin/$${targetPrefix}$${TARGET}$${targetSuffix}
            buildRootInstall.commands=mkdir $${FAKE_INSTALL} || true; \
            mkdir $${FAKE_INSTALL}/lib/ || true; \
            mkdir $${FAKE_INSTALL}/lib/akil/ || true; \
            cp -r $$postCommandBody $${FAKE_INSTALL}/lib/akil;
            QMAKE_POST_LINK+=$$buildRootInstall.commands
        }

        isEqual(TEMPLATE,app){
            message(cp -r $${OUT_PWD}/bin/$${TARGET} $$FAKE_INSTALL)
            buildRootInstall.commands=mkdir $${FAKE_INSTALL} || true; \
            mkdir $${FAKE_INSTALL}/bin/ || true; \
            mkdir $${FAKE_INSTALL}/bin/akil/ || true; \
            cp -r $${OUT_PWD}/bin/$${TARGET} $${FAKE_INSTALL}/bin/akil
            QMAKE_POST_LINK+=$$buildRootInstall.commands
        }
    }

    target.path = $${DEPL_ROOT}/bin/akil/
    INSTALLS +=  target
}
