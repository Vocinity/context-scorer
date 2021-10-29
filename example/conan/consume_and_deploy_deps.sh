
 #!/bin/bash
 # ./consumeAndDeployDeps.sh whereToPlaceEnv profileName whereToDeployDeps
 venvPlace=$1/conanEnv
 conanDir=`dirname "$0"`
 profileName=$2
 deployDir=$3/$profileName
 python3 -m venv $venvPlace || true
 source $venvPlace/bin/activate
 which python3
 #export CONAN_V2_MODE=1
 export CONAN_USER_HOME=$venvPlace/
 pip3 install wheel
 pip3 install conan
 if ! [[ `conan profile list` == *"$profileName"* ]]; then
 	echo "Creating new Conan profile '$profileName'"
 	conan profile new $profileName --detect || true
 	conan profile update settings.compiler.libcxx=libstdc++11 $profileName
 else
 	echo "Using already initialized profile $profileName"
 fi
 rm -rf $deployDir || true
 conan install $conanDir -pr=$profileName -s build_type=Release -g deploy --install-folder=$deployDir
 retVal=$?
 if [ $retVal -ne 0 ]; then
 	echo "Conan procedure hits the wall. See 'ERROR:' lines above."
 	exit $retVal
 fi
 subdircount=$(find $deployDir -maxdepth 1 -type d | wc -l)
 if [[ "$subdircount" -eq 1 ]]
 then
 	echo "Nothing to link in deploy directory. So nothing to tell linker."
 	exit 0
 fi
 LinkerPaths=$LD_LIBRARY_PATH
 for d in $deployDir/*/lib; do LinkerPaths="$LinkerPaths:$d"; done
 echo "Set your library search paths like below to let linker know about Conan in runtime."
 echo "export LD_LIBRARY_PATH=$LinkerPaths"
 echo "export LD_LIBRARY_PATH=$LinkerPaths" > $deployDir/../../share/akil/qmake/${profileName}-conan-linker-runtime.sh
 echo "Conan runtime linker instructions are also dumped to $deployDir/../../share/akil/qmake/${profileName}-conan-linker-runtime.sh"
