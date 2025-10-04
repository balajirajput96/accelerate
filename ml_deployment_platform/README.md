# 🚀 Complete ML Deployment Platform

A production-ready ML model deployment platform with interactive web interface, API endpoints, and automated deployment to multiple cloud platforms.

## ✨ Features

- **🎯 Interactive Web Dashboard** - Modern UI with real-time monitoring
- **🤖 Model Management** - Upload, deploy, and manage ML models
- **📊 Prediction Interface** - Single and batch predictions
- **🔬 API Testing** - Interactive API documentation
- **📈 Monitoring** - Performance metrics and charts
- **⚙️ Production Ready** - Docker, health checks, error handling

## 🚀 Quick Deploy (Choose One)

### 1. Railway (Fastest - 30 seconds)
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

### 2. Render (One-Click Deploy)
1. Go to [render.com](https://render.com)
2. Connect your GitHub repository
3. Select "Web Service"
4. Use the provided `render.yaml` configuration

### 3. Docker
```bash
# Build and run
docker build -t ml-deployment-platform .
docker run -d -p 8000:8000 ml-deployment-platform
```

### 4. Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 📋 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web Dashboard |
| `GET` | `/health` | Health check |
| `GET` | `/models` | List all models |
| `POST` | `/models/upload` | Upload new model |
| `POST` | `/predict` | Single prediction |
| `POST` | `/predict/batch` | Batch prediction |
| `DELETE` | `/models/{id}` | Delete model |

## 🎯 Usage Examples

### Upload Model
```bash
curl -X POST "http://localhost:8000/models/upload" \
  -F "file=@model.pkl" \
  -F "name=My Model" \
  -F "type=classification"
```

### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"model_id": "your-model-id", "features": [1.0, 2.0, 3.0]}'
```

### Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -F "model_id=your-model-id" \
  -F "file=@data.csv"
```

## 🛠️ Development

### Project Structure
```
ml_deployment_platform/
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
├── railway.json          # Railway deployment config
├── render.yaml           # Render deployment config
├── deploy.sh             # Automated deployment script
└── README.md             # This file
```

### Environment Variables
- `PORT` - Server port (default: 8000)
- `HOST` - Server host (default: 0.0.0.0)

## 🔧 Configuration

### Supported Model Formats
- `.pkl` - Pickle files
- `.joblib` - Joblib files
- `.h5` - Keras/TensorFlow models
- `.pth` - PyTorch models

### Supported Data Formats
- CSV files for batch predictions
- JSON for single predictions

## 📊 Monitoring

The platform includes built-in monitoring:
- Real-time health checks
- Model performance metrics
- Prediction history tracking
- Error logging and reporting

## 🚀 Production Deployment

### Railway (Recommended)
1. Install Railway CLI: `npm install -g @railway/cli`
2. Login: `railway login`
3. Deploy: `railway init && railway up`

### Render
1. Connect GitHub repository
2. Select "Web Service"
3. Use `render.yaml` configuration

### Docker
1. Build image: `docker build -t ml-platform .`
2. Run container: `docker run -p 8000:8000 ml-platform`

## 🎉 Your ML Platform is Ready!

Once deployed, you'll have:
- ✅ Interactive web dashboard
- ✅ Model upload and management
- ✅ Real-time predictions
- ✅ Batch processing
- ✅ API documentation
- ✅ Performance monitoring

**Access your platform at the provided URL and start deploying ML models!**

## 📞 Support

For issues or questions:
1. Check the health endpoint: `/health`
2. View API documentation: `/docs`
3. Check logs for error details

---

**Built with ❤️ using FastAPI, Python, and modern web technologies**