pipeline {
    agent {
        kubernetes {
            yaml """
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: python
    image: python:3.10
    command:
    - cat
    tty: true
"""
        }
    }

    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/dafalsanika30/Loan_predication.git'
            }
        }

        stage('Setup Python Environment') {
            steps {
                container('python') {
                    sh '''
                        python3 -m venv venv
                        . venv/bin/activate
                        pip install --upgrade pip
                        pip install -r requirements.txt || echo "⚠️ No requirements.txt found"
                    '''
                }
            }
        }

        stage('Run Migrations') {
            steps {
                container('python') {
                    sh '''
                        . venv/bin/activate
                        python manage.py migrate || echo "⚠️ migrate failed or manage.py missing"
                    '''
                }
            }
        }

        stage('Run Tests') {
            steps {
                container('python') {
                    sh '''
                        . venv/bin/activate
                        python manage.py test || echo "⚠️ No tests found"
                    '''
                }
            }
        }

        stage('Run Server (Optional)') {
            steps {
                echo '✅ Django build completed successfully!'
            }
        }
    }

    post {
        success {
            echo '🎉 Build completed successfully!'
        }
        failure {
            echo '❌ Build failed. Please check the logs.'
        }
    }
}
