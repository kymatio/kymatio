pipeline {
  agent none
  options {
    disableConcurrentBuilds()
    buildDiscarder(logRotator(numToKeepStr: '8', daysToKeepStr: '20'))
    timeout(time: 1, unit: 'HOURS')
  }
  stages {
    stage('main') {
      agent {
	dockerfile {
	  dir 'tools'
	  args '--gpus 1'
	}
      }
      environment {
	HOME = "$WORKSPACE/build"
      }
      steps {
	sh 'python3 -m pip --version'
	sh 'python3 -m pip freeze'
	sh 'python3 -c "import torch; torch.cuda.current_device()"'
	sh 'python3 -c "import tensorflow as tf; tf.test.is_gpu_available()"'
	sh 'python3 -m venv --system-site-packages --without-pip $HOME'
	sh '''#!/bin/bash -ex
	  source $HOME/bin/activate
	  python3 setup.py develop
	  TF_FORCE_GPU_ALLOW_GROWTH=true python3 -m pytest --cov=kymatio
	  python3 -m coverage xml
	  bash <(curl -s https://codecov.io/bash) -t 3941b784-370b-4e50-a162-e5018b7c2861 -F jenkins_$STAGE_NAME -s $WORKSPACE
	'''
      }
    }
  }
  post {
    failure {
      emailext subject: '$PROJECT_NAME - Build #$BUILD_NUMBER - $BUILD_STATUS',
	       body: '''$PROJECT_NAME - Build #$BUILD_NUMBER - $BUILD_STATUS

Check console output at $BUILD_URL to view full results.

Building $BRANCH_NAME for $CAUSE
$JOB_DESCRIPTION

Chages:
$CHANGES

End of build log:
${BUILD_LOG,maxLines=200}
''',
	       recipientProviders: [
		 [$class: 'DevelopersRecipientProvider'],
	       ],
	       replyTo: '$DEFAULT_REPLYTO',
	       to: 'janden@flatironinstitute.org'
    }
  }
}
