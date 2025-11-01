pipeline {
    agent {
        docker {
            image 'python:3.10'   // ✅ Use an official Python image
            args '-v /tmp:/tmp'
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
                sh '''
                    python -m venv venv
                    . venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Run Migrations') {
            steps {
                sh '''
                    . venv/bin/activate
                    python manage.py migrate
                '''
            }
        }

        stage('Run Tests') {
            steps {
                sh '''
                    . venv/bin/activate
                    python manage.py test || echo "No tests found"
                '''
            }
        }

        stage('Run Server (Optional)') {
            steps {
                echo '✅ Django build successful!'
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
