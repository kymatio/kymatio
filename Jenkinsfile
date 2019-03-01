def notifyFailure() {
  if (env.BRANCH_NAME.startsWith('PR-')) {
    gitHubPRStatus statusMessage: [content: '''
End of failed build log:
${BUILD_LOG,maxLines=200}
''']
  } else {
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

pipeline {
  agent none
  options {
    disableConcurrentBuilds()
    buildDiscarder(logRotator(numToKeepStr: '8', daysToKeepStr: '20'))
    timeout(time: 1, unit: 'HOURS')
  }
  stages {
    stage('torch') {
      agent {
	dockerfile {
	  dir 'tools'
	  args '--device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm'
	}
      }
      environment {
	HOME = pwd(tmp:true)
      }
      steps {
	sh 'python3 -m venv $HOME'
	sh '''#!/bin/bash -ex
	  source $HOME/bin/activate
	  python3 setup.py develop
	  KYMATIO_BACKEND=$STAGE_NAME pytest --cov=kymatio
	  bash <(curl -s https://codecov.io/bash) -t 3941b784-370b-4e50-a162-e5018b7c2861 -F jenkins_$STAGE_NAME
	'''
      }
    }
    stage('skcuda') {
      agent {
	dockerfile {
	  dir 'tools'
	  args '--device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm'
	}
      }
      environment {
	HOME = pwd(tmp:true)
      }
      steps {
	sh 'python3 -m venv $HOME'
	sh '''#!/bin/bash -ex
	  source $HOME/bin/activate
	  python3 setup.py develop
	  KYMATIO_BACKEND=$STAGE_NAME pytest --cov=kymatio
	  bash <(curl -s https://codecov.io/bash) -t 3941b784-370b-4e50-a162-e5018b7c2861 -F jenkins_$STAGE_NAME
	'''
      }
    }
  }
  post {
    failure {
      notifyFailure()
    }
  }
}
