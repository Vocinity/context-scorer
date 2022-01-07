include(config.pri)
include(dependency.pri)
include(deploy.pri)

DISTFILES += \
    $$PWD/depend_context-scorer_template.pri \
#    ../.gitlab-ci.yml  \
    ../conan/consume_and_deploy_deps.sh \
    ../scorer.py \
    ../doc/Doxyfile

# bear -o %{sourceDir}/compile_commands.json make %{buildDir}
#TIDY_CHECKS = -checks=-*,modernize-*,clang-analyzer-*,bugprone-*,cert-*,cppcoreguidelines-*,hicpp-*,misc-*,performance-*,readability-*
#system(/usr/bin/run-clang-tidy.py $$TIDY_CHECKS -j 8 2>&1 | sed -r \"s/\\(([0-9]+)\\)/:\\1/g\" 1>&2)
