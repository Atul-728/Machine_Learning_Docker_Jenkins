pipeline {
    agent any
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Build Docker Image') {
            steps {
                bat 'docker build -t milk-quality-predictor:%BUILD_NUMBER% .'
                bat 'docker tag milk-quality-predictor:%BUILD_NUMBER% milk-quality-predictor:latest'
            }
        }
        
        stage('Run Tests') {
            steps {
                bat 'docker run --rm milk-quality-predictor:latest python -m unittest discover -s tests'
            }
        }
        
        stage('Deploy to Development') {
            when {
                branch 'develop'
            }
            steps {
                bat 'docker-compose down || true'
                bat 'docker-compose up -d'
                echo 'Deployed to development environment'
            }
        }
        
        stage('Deploy to Production') {
            when {
                branch 'main'
            }
            steps {
                input message: 'Deploy to production?', ok: 'Yes'
                bat 'docker stop milk-quality-app || true'
                bat 'docker rm milk-quality-app || true'
                bat 'docker run -d --name milk-quality-app -p 80:5000 milk-quality-predictor:latest'
                echo 'Deployed to production environment'
            }
        }
    }
    
    post {
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed!'
        }
    }
}
