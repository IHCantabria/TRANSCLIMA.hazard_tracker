#!/bin/bash

################################################################################
#                                                                              # 
#                       DATOS A MODIFICAR POR EL USUARIO                       #

export PROJECT_KEY=IHCantabria_template.python.lib_3a6e6e1c-9615-45f0-b54f-db978ffc9844
export SONAR_TOKEN="sqp_897eb68d760497412637c770f1ae085728e44621"

#                                                                              #
################################################################################

export SONAR_SCANNER_VERSION=5.0.1.3006
export SONAR_SCANNER_HOME=$HOME/.sonar/sonar-scanner-$SONAR_SCANNER_VERSION-linux
if [ ! -d $SONAR_SCANNER_HOME/bin ]; then
  curl --create-dirs -sSLo $HOME/.sonar/sonar-scanner.zip https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-$SONAR_SCANNER_VERSION-linux.zip
  unzip -o $HOME/.sonar/sonar-scanner.zip -d $HOME/.sonar/
  rm $HOME/.sonar/sonar-scanner.zip
fi
export PATH=$SONAR_SCANNER_HOME/bin:$PATH
export SONAR_SCANNER_OPTS="-server"

sonar-scanner \
  -Dsonar.projectKey=$PROJECT_KEY \
  -Dsonar.sources=. \
  -Dsonar.host.url=http://ihsonarqube.ihcantabria.com:9000

cat .scannerwork/report-task.txt
rm -rf .scannerwork