DISTFILES+= $$PWD/../conan/consume_and_deploy_deps.sh \
    $$PWD/../conan/conanfile.txt \
    $$PWD/../conan/consume_and_deploy_deps.sh

unix{
    !NO_CONAN{
        if(exists($$PWD/../conan/conanfile.py)|exists($$PWD/../conan/conanfile.txt)){
           HasPython=false
           system("python3 --version"): HasPython = true
           if($$HasPython){
                if(!exists($$PWD/../conan/consume_and_deploy_deps.sh)){
                    if(!exists($$PWD/../conan)){
                        system(mkdir $$PWD/../conan)
                    }
                    ScriptTemplate='$$escape_expand(\\n)\
                    $${LITERAL_HASH}!/bin/bash$$escape_expand(\\n)\
                    $${LITERAL_HASH} ./consume_and_deploy_deps.sh whereToPlaceEnv profileName whereToDeployDeps$$escape_expand(\\n)\
                    venvPlace=$1/conanEnv$$escape_expand(\\n)\
                    conanDir=`dirname \"$0\"`$$escape_expand(\\n)\
                    profileName=$2$$escape_expand(\\n)\
                    deployDir=$3/$profileName$$escape_expand(\\n)\
                    python3 -m venv $venvPlace || true$$escape_expand(\\n)\
                    source $venvPlace/bin/activate$$escape_expand(\\n)\
                    which python3$$escape_expand(\\n)\
                    $${LITERAL_HASH}export CONAN_V2_MODE=1$$escape_expand(\\n)\
                    export CONAN_USER_HOME=$venvPlace/$$escape_expand(\\n)\
                    pip3 install wheel$$escape_expand(\\n)\
                    pip3 install conan$$escape_expand(\\n)\
                    if ! [[ `conan profile list` == *\"$profileName\"* ]]; then$$escape_expand(\\n)\
                    $$escape_expand(\\t)echo \"Creating new Conan profile \'$profileName\'\"$$escape_expand(\\n)\
                    $$escape_expand(\\t)conan profile new $profileName --detect || true$$escape_expand(\\n)\
                    $$escape_expand(\\t)conan profile update settings.compiler.libcxx=libstdc++11 $profileName$$escape_expand(\\n)\
                    else$$escape_expand(\\n)\
                    $$escape_expand(\\t)echo \"Using already initialized profile $profileName\"$$escape_expand(\\n)\
                    fi$$escape_expand(\\n)\
                    rm -rf $deployDir || true$$escape_expand(\\n)\
                    conan install $conanDir -pr=$profileName -s build_type=Release -g deploy --install-folder=$deployDir$$escape_expand(\\n)\
                    retVal=$?$$escape_expand(\\n)\
                    if [ $retVal -ne 0 ]; then$$escape_expand(\\n)\
                    $$escape_expand(\\t)echo \"Conan procedure hits the wall. See \'ERROR:\' lines above.\"$$escape_expand(\\n)\
                    $$escape_expand(\\t)exit $retVal$$escape_expand(\\n)\
                    fi$$escape_expand(\\n)\
                    subdircount=$(find $deployDir -maxdepth 1 -type d | wc -l)$$escape_expand(\\n)\
                    if [[ \"$subdircount\" -eq 1 ]]$$escape_expand(\\n)\
                    then$$escape_expand(\\n)\
                    $$escape_expand(\\t)echo \"Nothing to link in deploy directory. So nothing to tell linker.\"$$escape_expand(\\n)\
                    $$escape_expand(\\t)exit 0$$escape_expand(\\n)\
                    fi$$escape_expand(\\n)\
                    LinkerPaths=$LD_LIBRARY_PATH$$escape_expand(\\n)\
                    for d in $deployDir/*/lib; do LinkerPaths=\"$LinkerPaths:$d\"; done$$escape_expand(\\n)\
                    echo \"Set your library search paths like below to let linker know about Conan in runtime.\"$$escape_expand(\\n)\
                    echo \"export LD_LIBRARY_PATH=$LinkerPaths\"$$escape_expand(\\n)\
                    echo \"export LD_LIBRARY_PATH=$LinkerPaths\" > $deployDir/../../share/akil/qmake/${profileName}-conan-linker-runtime.sh$$escape_expand(\\n)\
                    echo \"Conan runtime linker instructions are also dumped to $deployDir/../../share/akil/qmake/${profileName}-conan-linker-runtime.sh\"'
                    message("Exporting Conan helper $$PWD/../conan/consume_and_deploy_deps.sh")
                    write_file($$PWD/../conan/consume_and_deploy_deps.sh,ScriptTemplate)
                }
                message("Conan Package Management in use.")
                CONFIG+=ON_CONAN
            }else{
                error("You need to get python3 dev package for setting up Conan workspace. E.g. 'sudo apt install libpython3-dev'.")
            }
        }else{
            error("Either $$PWD/../conan/conanfile.txt or $$PWD/../conan/conanfile.py Conan recipe is needed to utilize Conan.")
        }
    }else{
        warning("Conan is off. You need to satisfy missing requirements yourself!")
    }

    ON_CONAN{
        ConanDeployDir=$${DEPL_ROOT}/conan/
        isEmpty(PROJECT_NAME){
            PROJECT_NAME=$$TARGET
        }
        message("$$PROJECT_NAME Conan profile is being processed.")

        ConanPri=$${ConanDeployDir}/$${PROJECT_NAME}/conanbuildinfo.pri
        message("Checking $${ConanPri}")
        if(exists($${ConanPri})){
            message("Using Conan workspace in $${ConanDeployDir}")
            prebuild.commands = /bin/bash $$PWD/../conan/consume_and_deploy_deps.sh $$OUT_PWD/../ $$PROJECT_NAME $${ConanDeployDir}
            first.depends = prebuild
            QMAKE_EXTRA_TARGETS += prebuild first
        }else{
            !build_pass{
            ConanDone=-2
                ConanMessage = $$system("/bin/bash $$PWD/../conan/consume_and_deploy_deps.sh $$OUT_PWD/../ $$PROJECT_NAME $${ConanDeployDir}",false,ConanDone)
                equals(ConanDone,0){
                    message($${ConanMessage})
                    message("Conan workspace initialized.")
                }else{
                    error($${ConanMessage})
                    message("Conan Initialization FAILED.")
                }
            }
        }
        CONFIG += conan_basic_setup
        !build_pass:include($${ConanPri})
        ConanClean.commands = rm -r $${ConanDeployDir}/$${PROJECT_NAME} $$OUT_PWD/../conanEnv $${ConanDeployDir}/../share/akil/qmake/$${PROJECT_NAME}-conan-linker-runtime.sh
        distclean.depends += ConanClean
        QMAKE_EXTRA_TARGETS += distclean ConanClean
    }
}
