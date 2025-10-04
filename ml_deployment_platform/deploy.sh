#!/bin/bash

# 🚀 Complete ML Deployment Platform - Automated Deployment Script
# This script handles deployment to multiple platforms

set -e

echo "🚀 Starting ML Deployment Platform Setup..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.8+ first."
        exit 1
    fi
    print_success "Python 3 found: $(python3 --version)"
}

# Install dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    pip3 install -r requirements.txt
    print_success "Dependencies installed successfully"
}

# Test the application locally
test_local() {
    print_status "Testing application locally..."
    
    # Start the application in background
    python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 &
    APP_PID=$!
    
    # Wait for app to start
    sleep 5
    
    # Test health endpoint
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_success "Local test passed - API is responding"
    else
        print_error "Local test failed - API not responding"
        kill $APP_PID 2>/dev/null || true
        exit 1
    fi
    
    # Stop the application
    kill $APP_PID 2>/dev/null || true
    print_success "Local testing completed"
}

# Deploy to Railway
deploy_railway() {
    print_status "Deploying to Railway..."
    
    # Check if Railway CLI is installed
    if ! command -v railway &> /dev/null; then
        print_status "Installing Railway CLI..."
        npm install -g @railway/cli
    fi
    
    # Login to Railway
    print_status "Logging into Railway..."
    railway login
    
    # Initialize and deploy
    print_status "Initializing Railway project..."
    railway init
    
    print_status "Deploying to Railway..."
    railway up
    
    print_success "Railway deployment initiated! Check your Railway dashboard for the live URL."
}

# Deploy to Render
deploy_render() {
    print_status "Preparing for Render deployment..."
    print_status "Render configuration created in render.yaml"
    print_warning "To deploy to Render:"
    echo "1. Go to https://render.com"
    echo "2. Connect your GitHub repository"
    echo "3. Select 'Web Service'"
    echo "4. Use the render.yaml configuration"
    print_success "Render deployment configuration ready!"
}

# Deploy with Docker
deploy_docker() {
    print_status "Building Docker image..."
    docker build -t ml-deployment-platform .
    
    print_status "Running Docker container..."
    docker run -d -p 8000:8000 --name ml-platform ml-deployment-platform
    
    print_success "Docker deployment completed!"
    print_status "Access your application at: http://localhost:8000"
}

# Create deployment instructions
create_instructions() {
    cat > DEPLOYMENT_INSTRUCTIONS.md << EOF
# 🚀 ML Deployment Platform - Deployment Instructions

## Quick Deploy Options

### 1. Railway (Recommended - Fastest)
\`\`\`bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
\`\`\`

### 2. Render (One-Click Deploy)
1. Go to https://render.com
2. Connect your GitHub repository
3. Select "Web Service"
4. Use the provided render.yaml configuration

### 3. Docker
\`\`\`bash
# Build and run
docker build -t ml-deployment-platform .
docker run -d -p 8000:8000 ml-deployment-platform
\`\`\`

### 4. Local Development
\`\`\`bash
# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn main:app --host 0.0.0.0 --port 8000
\`\`\`

## Features Included

✅ **Interactive Web Dashboard** - Modern UI with real-time monitoring
✅ **Model Management** - Upload, deploy, and manage ML models
✅ **Prediction API** - Single and batch predictions
✅ **Real-time Monitoring** - Performance metrics and charts
✅ **API Documentation** - Interactive API testing
✅ **Production Ready** - Docker, health checks, error handling

## API Endpoints

- \`GET /\` - Web Dashboard
- \`GET /health\` - Health check
- \`GET /models\` - List models
- \`POST /models/upload\` - Upload model
- \`POST /predict\` - Single prediction
- \`POST /predict/batch\` - Batch prediction
- \`DELETE /models/{id}\` - Delete model

## Usage

1. **Upload Models**: Use the web interface or API to upload .pkl, .joblib, .h5, .pth files
2. **Make Predictions**: Use the prediction interface or API endpoints
3. **Monitor Performance**: View real-time metrics and prediction history
4. **API Integration**: Use the REST API for integration with other systems

Your ML deployment platform is ready to use! 🎉
EOF

    print_success "Deployment instructions created in DEPLOYMENT_INSTRUCTIONS.md"
}

# Main deployment function
main() {
    echo "🎉 ML Deployment Platform - Complete Setup"
    echo "=========================================="
    
    # Check prerequisites
    check_python
    
    # Install dependencies
    install_dependencies
    
    # Test locally
    test_local
    
    # Create deployment instructions
    create_instructions
    
    echo ""
    echo "🎉 DEPLOYMENT READY!"
    echo "==================="
    echo ""
    echo "Choose your deployment method:"
    echo ""
    echo "1. Railway (Fastest - 30 seconds)"
    echo "   Run: railway login && railway init && railway up"
    echo ""
    echo "2. Render (One-click)"
    echo "   Go to: https://render.com and connect your repo"
    echo ""
    echo "3. Docker"
    echo "   Run: docker build -t ml-platform . && docker run -p 8000:8000 ml-platform"
    echo ""
    echo "4. Local Development"
    echo "   Run: uvicorn main:app --host 0.0.0.0 --port 8000"
    echo ""
    echo "📖 See DEPLOYMENT_INSTRUCTIONS.md for detailed steps"
    echo ""
    print_success "Your complete ML deployment platform is ready! 🚀"
}

# Run main function
main "$@"