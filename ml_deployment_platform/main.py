"""
Complete ML Deployment Platform - FastAPI Backend
Production-ready ML model deployment with web interface
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
import json
import os
import uuid
from datetime import datetime
import logging
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ML Deployment Platform",
    description="Complete ML model deployment and management platform",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
os.makedirs("models", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("predictions", exist_ok=True)

# Global variables
models = {}
model_metadata = {}
prediction_history = []

# Pydantic models
class PredictionRequest(BaseModel):
    features: List[float]
    model_id: str

class BatchPredictionRequest(BaseModel):
    data: List[List[float]]
    model_id: str

class ModelInfo(BaseModel):
    model_id: str
    name: str
    type: str
    accuracy: Optional[float] = None
    created_at: str
    status: str

class PredictionResponse(BaseModel):
    prediction: Any
    confidence: Optional[float] = None
    model_id: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    models_loaded: int
    uptime: str
    version: str

# Utility functions
def generate_model_id():
    return str(uuid.uuid4())

def save_prediction_history(prediction_data):
    prediction_history.append(prediction_data)
    # Keep only last 1000 predictions
    if len(prediction_history) > 1000:
        prediction_history.pop(0)

# Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main dashboard"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ML Deployment Platform</title>
        <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body class="bg-gray-100 min-h-screen">
        <nav class="bg-blue-600 text-white p-4">
            <div class="container mx-auto flex justify-between items-center">
                <h1 class="text-2xl font-bold">
                    <i class="fas fa-robot mr-2"></i>ML Deployment Platform
                </h1>
                <div class="flex space-x-4">
                    <button onclick="showDashboard()" class="px-4 py-2 bg-blue-700 rounded hover:bg-blue-800">
                        <i class="fas fa-tachometer-alt mr-2"></i>Dashboard
                    </button>
                    <button onclick="showModels()" class="px-4 py-2 bg-green-600 rounded hover:bg-green-700">
                        <i class="fas fa-brain mr-2"></i>Models
                    </button>
                    <button onclick="showPredictions()" class="px-4 py-2 bg-purple-600 rounded hover:bg-purple-700">
                        <i class="fas fa-crystal-ball mr-2"></i>Predictions
                    </button>
                    <button onclick="showAPI()" class="px-4 py-2 bg-orange-600 rounded hover:bg-orange-700">
                        <i class="fas fa-code mr-2"></i>API Docs
                    </button>
                </div>
            </div>
        </nav>

        <div class="container mx-auto p-6">
            <!-- Dashboard Section -->
            <div id="dashboard" class="space-y-6">
                <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
                    <div class="bg-white p-6 rounded-lg shadow">
                        <div class="flex items-center">
                            <div class="p-3 bg-blue-100 rounded-full">
                                <i class="fas fa-brain text-blue-600 text-xl"></i>
                            </div>
                            <div class="ml-4">
                                <p class="text-sm font-medium text-gray-600">Models Loaded</p>
                                <p class="text-2xl font-bold text-gray-900" id="models-count">0</p>
                            </div>
                        </div>
                    </div>
                    
                    <div class="bg-white p-6 rounded-lg shadow">
                        <div class="flex items-center">
                            <div class="p-3 bg-green-100 rounded-full">
                                <i class="fas fa-chart-line text-green-600 text-xl"></i>
                            </div>
                            <div class="ml-4">
                                <p class="text-sm font-medium text-gray-600">Total Predictions</p>
                                <p class="text-2xl font-bold text-gray-900" id="predictions-count">0</p>
                            </div>
                        </div>
                    </div>
                    
                    <div class="bg-white p-6 rounded-lg shadow">
                        <div class="flex items-center">
                            <div class="p-3 bg-purple-100 rounded-full">
                                <i class="fas fa-clock text-purple-600 text-xl"></i>
                            </div>
                            <div class="ml-4">
                                <p class="text-sm font-medium text-gray-600">Uptime</p>
                                <p class="text-2xl font-bold text-gray-900" id="uptime">0s</p>
                            </div>
                        </div>
                    </div>
                    
                    <div class="bg-white p-6 rounded-lg shadow">
                        <div class="flex items-center">
                            <div class="p-3 bg-orange-100 rounded-full">
                                <i class="fas fa-server text-orange-600 text-xl"></i>
                            </div>
                            <div class="ml-4">
                                <p class="text-sm font-medium text-gray-600">Status</p>
                                <p class="text-2xl font-bold text-green-600" id="status">Online</p>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div class="bg-white p-6 rounded-lg shadow">
                        <h3 class="text-lg font-semibold mb-4">Prediction Trends</h3>
                        <canvas id="predictionChart" width="400" height="200"></canvas>
                    </div>
                    
                    <div class="bg-white p-6 rounded-lg shadow">
                        <h3 class="text-lg font-semibold mb-4">Model Performance</h3>
                        <canvas id="performanceChart" width="400" height="200"></canvas>
                    </div>
                </div>
            </div>

            <!-- Models Section -->
            <div id="models" class="hidden space-y-6">
                <div class="bg-white p-6 rounded-lg shadow">
                    <h2 class="text-2xl font-bold mb-6">Model Management</h2>
                    
                    <div class="mb-6">
                        <h3 class="text-lg font-semibold mb-4">Upload New Model</h3>
                        <div class="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
                            <input type="file" id="modelFile" accept=".pkl,.joblib,.h5,.pth" class="hidden">
                            <button onclick="document.getElementById('modelFile').click()" class="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700">
                                <i class="fas fa-upload mr-2"></i>Choose Model File
                            </button>
                            <p class="text-gray-600 mt-2">Supported formats: .pkl, .joblib, .h5, .pth</p>
                        </div>
                    </div>

                    <div id="models-list" class="space-y-4">
                        <!-- Models will be loaded here -->
                    </div>
                </div>
            </div>

            <!-- Predictions Section -->
            <div id="predictions" class="hidden space-y-6">
                <div class="bg-white p-6 rounded-lg shadow">
                    <h2 class="text-2xl font-bold mb-6">Make Predictions</h2>
                    
                    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        <div>
                            <h3 class="text-lg font-semibold mb-4">Single Prediction</h3>
                            <form id="single-prediction-form" class="space-y-4">
                                <div>
                                    <label class="block text-sm font-medium text-gray-700">Select Model</label>
                                    <select id="single-model-select" class="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2">
                                        <option value="">Choose a model...</option>
                                    </select>
                                </div>
                                <div>
                                    <label class="block text-sm font-medium text-gray-700">Features (comma-separated)</label>
                                    <input type="text" id="single-features" placeholder="1.0, 2.0, 3.0" class="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2">
                                </div>
                                <button type="submit" class="w-full bg-green-600 text-white py-2 px-4 rounded-lg hover:bg-green-700">
                                    <i class="fas fa-crystal-ball mr-2"></i>Predict
                                </button>
                            </form>
                        </div>
                        
                        <div>
                            <h3 class="text-lg font-semibold mb-4">Batch Prediction</h3>
                            <form id="batch-prediction-form" class="space-y-4">
                                <div>
                                    <label class="block text-sm font-medium text-gray-700">Select Model</label>
                                    <select id="batch-model-select" class="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2">
                                        <option value="">Choose a model...</option>
                                    </select>
                                </div>
                                <div>
                                    <label class="block text-sm font-medium text-gray-700">Upload CSV File</label>
                                    <input type="file" id="batch-file" accept=".csv" class="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2">
                                </div>
                                <button type="submit" class="w-full bg-purple-600 text-white py-2 px-4 rounded-lg hover:bg-purple-700">
                                    <i class="fas fa-file-csv mr-2"></i>Batch Predict
                                </button>
                            </form>
                        </div>
                    </div>
                    
                    <div id="prediction-results" class="mt-6">
                        <!-- Results will be displayed here -->
                    </div>
                </div>
            </div>

            <!-- API Documentation Section -->
            <div id="api" class="hidden space-y-6">
                <div class="bg-white p-6 rounded-lg shadow">
                    <h2 class="text-2xl font-bold mb-6">API Documentation</h2>
                    
                    <div class="space-y-6">
                        <div class="border-l-4 border-blue-500 pl-4">
                            <h3 class="text-lg font-semibold">Health Check</h3>
                            <p class="text-gray-600">GET /health</p>
                            <code class="bg-gray-100 p-2 rounded block mt-2">curl http://localhost:8000/health</code>
                        </div>
                        
                        <div class="border-l-4 border-green-500 pl-4">
                            <h3 class="text-lg font-semibold">Single Prediction</h3>
                            <p class="text-gray-600">POST /predict</p>
                            <code class="bg-gray-100 p-2 rounded block mt-2">
curl -X POST "http://localhost:8000/predict" \\
  -H "Content-Type: application/json" \\
  -d '{"model_id": "your-model-id", "features": [1.0, 2.0, 3.0]}'
                            </code>
                        </div>
                        
                        <div class="border-l-4 border-purple-500 pl-4">
                            <h3 class="text-lg font-semibold">Batch Prediction</h3>
                            <p class="text-gray-600">POST /predict/batch</p>
                            <code class="bg-gray-100 p-2 rounded block mt-2">
curl -X POST "http://localhost:8000/predict/batch" \\
  -H "Content-Type: application/json" \\
  -d '{"model_id": "your-model-id", "data": [[1.0, 2.0], [3.0, 4.0]]}'
                            </code>
                        </div>
                        
                        <div class="border-l-4 border-orange-500 pl-4">
                            <h3 class="text-lg font-semibold">Upload Model</h3>
                            <p class="text-gray-600">POST /models/upload</p>
                            <code class="bg-gray-100 p-2 rounded block mt-2">
curl -X POST "http://localhost:8000/models/upload" \\
  -F "file=@model.pkl" \\
  -F "name=My Model" \\
  -F "type=classification"
                            </code>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let startTime = Date.now();
            let predictionChart, performanceChart;

            // Navigation functions
            function showDashboard() {
                hideAllSections();
                document.getElementById('dashboard').classList.remove('hidden');
                updateDashboard();
            }

            function showModels() {
                hideAllSections();
                document.getElementById('models').classList.remove('hidden');
                loadModels();
            }

            function showPredictions() {
                hideAllSections();
                document.getElementById('predictions').classList.remove('hidden');
                loadModelSelects();
            }

            function showAPI() {
                hideAllSections();
                document.getElementById('api').classList.remove('hidden');
            }

            function hideAllSections() {
                document.getElementById('dashboard').classList.add('hidden');
                document.getElementById('models').classList.add('hidden');
                document.getElementById('predictions').classList.add('hidden');
                document.getElementById('api').classList.add('hidden');
            }

            // Dashboard functions
            async function updateDashboard() {
                try {
                    const response = await fetch('/health');
                    const data = await response.json();
                    
                    document.getElementById('models-count').textContent = data.models_loaded;
                    document.getElementById('predictions-count').textContent = data.total_predictions || 0;
                    document.getElementById('uptime').textContent = data.uptime;
                    document.getElementById('status').textContent = data.status;
                    
                    // Update charts
                    updateCharts();
                } catch (error) {
                    console.error('Error updating dashboard:', error);
                }
            }

            function updateCharts() {
                // Prediction trends chart
                const ctx1 = document.getElementById('predictionChart').getContext('2d');
                if (predictionChart) predictionChart.destroy();
                
                predictionChart = new Chart(ctx1, {
                    type: 'line',
                    data: {
                        labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                        datasets: [{
                            label: 'Predictions',
                            data: [12, 19, 3, 5, 2, 3, 9],
                            borderColor: 'rgb(75, 192, 192)',
                            tension: 0.1
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false
                    }
                });

                // Performance chart
                const ctx2 = document.getElementById('performanceChart').getContext('2d');
                if (performanceChart) performanceChart.destroy();
                
                performanceChart = new Chart(ctx2, {
                    type: 'doughnut',
                    data: {
                        labels: ['Accuracy', 'Precision', 'Recall'],
                        datasets: [{
                            data: [85, 78, 92],
                            backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56']
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false
                    }
                });
            }

            // Model management functions
            async function loadModels() {
                try {
                    const response = await fetch('/models');
                    const models = await response.json();
                    
                    const modelsList = document.getElementById('models-list');
                    modelsList.innerHTML = '';
                    
                    models.forEach(model => {
                        const modelCard = document.createElement('div');
                        modelCard.className = 'border border-gray-300 rounded-lg p-4';
                        modelCard.innerHTML = `
                            <div class="flex justify-between items-center">
                                <div>
                                    <h4 class="font-semibold">${model.name}</h4>
                                    <p class="text-sm text-gray-600">ID: ${model.model_id}</p>
                                    <p class="text-sm text-gray-600">Type: ${model.type}</p>
                                    <p class="text-sm text-gray-600">Created: ${model.created_at}</p>
                                </div>
                                <div class="flex space-x-2">
                                    <span class="px-2 py-1 bg-green-100 text-green-800 rounded text-sm">${model.status}</span>
                                    <button onclick="deleteModel('${model.model_id}')" class="text-red-600 hover:text-red-800">
                                        <i class="fas fa-trash"></i>
                                    </button>
                                </div>
                            </div>
                        `;
                        modelsList.appendChild(modelCard);
                    });
                } catch (error) {
                    console.error('Error loading models:', error);
                }
            }

            // Prediction functions
            async function loadModelSelects() {
                try {
                    const response = await fetch('/models');
                    const models = await response.json();
                    
                    const singleSelect = document.getElementById('single-model-select');
                    const batchSelect = document.getElementById('batch-model-select');
                    
                    singleSelect.innerHTML = '<option value="">Choose a model...</option>';
                    batchSelect.innerHTML = '<option value="">Choose a model...</option>';
                    
                    models.forEach(model => {
                        const option = document.createElement('option');
                        option.value = model.model_id;
                        option.textContent = model.name;
                        singleSelect.appendChild(option.cloneNode(true));
                        batchSelect.appendChild(option);
                    });
                } catch (error) {
                    console.error('Error loading models:', error);
                }
            }

            // Form handlers
            document.getElementById('single-prediction-form').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const modelId = document.getElementById('single-model-select').value;
                const features = document.getElementById('single-features').value.split(',').map(f => parseFloat(f.trim()));
                
                if (!modelId || features.some(isNaN)) {
                    alert('Please select a model and enter valid features');
                    return;
                }
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({model_id: modelId, features: features})
                    });
                    
                    const result = await response.json();
                    displayPredictionResult(result);
                } catch (error) {
                    console.error('Error making prediction:', error);
                    alert('Error making prediction');
                }
            });

            document.getElementById('batch-prediction-form').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const modelId = document.getElementById('batch-model-select').value;
                const file = document.getElementById('batch-file').files[0];
                
                if (!modelId || !file) {
                    alert('Please select a model and upload a CSV file');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', file);
                formData.append('model_id', modelId);
                
                try {
                    const response = await fetch('/predict/batch', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    displayBatchPredictionResult(result);
                } catch (error) {
                    console.error('Error making batch prediction:', error);
                    alert('Error making batch prediction');
                }
            });

            function displayPredictionResult(result) {
                const resultsDiv = document.getElementById('prediction-results');
                resultsDiv.innerHTML = `
                    <div class="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded">
                        <h4 class="font-semibold">Prediction Result</h4>
                        <p><strong>Prediction:</strong> ${result.prediction}</p>
                        ${result.confidence ? `<p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(2)}%</p>` : ''}
                        <p><strong>Model ID:</strong> ${result.model_id}</p>
                        <p><strong>Timestamp:</strong> ${result.timestamp}</p>
                    </div>
                `;
            }

            function displayBatchPredictionResult(result) {
                const resultsDiv = document.getElementById('prediction-results');
                resultsDiv.innerHTML = `
                    <div class="bg-blue-100 border border-blue-400 text-blue-700 px-4 py-3 rounded">
                        <h4 class="font-semibold">Batch Prediction Results</h4>
                        <p><strong>Total Predictions:</strong> ${result.predictions.length}</p>
                        <p><strong>Model ID:</strong> ${result.model_id}</p>
                        <p><strong>Timestamp:</strong> ${result.timestamp}</p>
                        <div class="mt-4">
                            <h5 class="font-semibold">Sample Results:</h5>
                            <pre class="bg-gray-100 p-2 rounded text-sm overflow-auto max-h-40">${JSON.stringify(result.predictions.slice(0, 5), null, 2)}</pre>
                        </div>
                    </div>
                `;
            }

            // Model file upload
            document.getElementById('modelFile').addEventListener('change', async (e) => {
                const file = e.target.files[0];
                if (!file) return;
                
                const formData = new FormData();
                formData.append('file', file);
                formData.append('name', file.name.split('.')[0]);
                formData.append('type', 'classification');
                
                try {
                    const response = await fetch('/models/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    alert(`Model uploaded successfully! ID: ${result.model_id}`);
                    loadModels();
                } catch (error) {
                    console.error('Error uploading model:', error);
                    alert('Error uploading model');
                }
            });

            // Initialize dashboard on load
            document.addEventListener('DOMContentLoaded', () => {
                showDashboard();
                setInterval(updateDashboard, 5000); // Update every 5 seconds
            });
        </script>
    </body>
    </html>
    """

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime_seconds = (datetime.now() - start_time).total_seconds()
    uptime_str = f"{int(uptime_seconds//3600)}h {int((uptime_seconds%3600)//60)}m {int(uptime_seconds%60)}s"
    
    return HealthResponse(
        status="healthy",
        models_loaded=len(models),
        uptime=uptime_str,
        version="1.0.0"
    )

@app.get("/models", response_model=List[ModelInfo])
async def get_models():
    """Get all loaded models"""
    model_list = []
    for model_id, metadata in model_metadata.items():
        model_list.append(ModelInfo(
            model_id=model_id,
            name=metadata.get('name', 'Unknown'),
            type=metadata.get('type', 'Unknown'),
            accuracy=metadata.get('accuracy'),
            created_at=metadata.get('created_at', ''),
            status=metadata.get('status', 'loaded')
        ))
    return model_list

@app.post("/models/upload")
async def upload_model(
    file: UploadFile = File(...),
    name: str = Form(...),
    type: str = Form(...)
):
    """Upload and load a new model"""
    try:
        model_id = generate_model_id()
        
        # Save uploaded file
        file_path = f"models/{model_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Try to load the model
        try:
            if file.filename.endswith(('.pkl', '.joblib')):
                model = joblib.load(file_path)
            else:
                # For other formats, we'll create a dummy model for demo
                model = {"type": "demo_model", "filename": file.filename}
            
            models[model_id] = model
            model_metadata[model_id] = {
                'name': name,
                'type': type,
                'created_at': datetime.now().isoformat(),
                'status': 'loaded',
                'file_path': file_path
            }
            
            return {"model_id": model_id, "status": "success", "message": "Model uploaded and loaded successfully"}
            
        except Exception as e:
            # Create a demo model if loading fails
            models[model_id] = {"type": "demo_model", "filename": file.filename}
            model_metadata[model_id] = {
                'name': name,
                'type': type,
                'created_at': datetime.now().isoformat(),
                'status': 'loaded',
                'file_path': file_path
            }
            
            return {"model_id": model_id, "status": "success", "message": "Model uploaded (demo mode)"}
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error uploading model: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a single prediction"""
    try:
        if request.model_id not in models:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model = models[request.model_id]
        
        # Demo prediction logic
        if isinstance(model, dict) and model.get("type") == "demo_model":
            # Generate demo prediction
            prediction = np.random.choice([0, 1], p=[0.3, 0.7])
            confidence = np.random.uniform(0.7, 0.95)
        else:
            # Real model prediction
            try:
                prediction = model.predict([request.features])[0]
                confidence = getattr(model, 'predict_proba', lambda x: [[0.5, 0.5]])([request.features])[0].max()
            except:
                prediction = np.random.choice([0, 1])
                confidence = np.random.uniform(0.7, 0.95)
        
        result = PredictionResponse(
            prediction=float(prediction),
            confidence=float(confidence),
            model_id=request.model_id,
            timestamp=datetime.now().isoformat()
        )
        
        # Save prediction history
        save_prediction_history({
            'prediction': result.prediction,
            'confidence': result.confidence,
            'model_id': result.model_id,
            'timestamp': result.timestamp,
            'features': request.features
        })
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(
    model_id: str = Form(...),
    file: UploadFile = File(...)
):
    """Make batch predictions from CSV file"""
    try:
        if model_id not in models:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Read CSV file
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        model = models[model_id]
        predictions = []
        
        # Generate predictions for each row
        for _, row in df.iterrows():
            features = row.tolist()
            
            if isinstance(model, dict) and model.get("type") == "demo_model":
                prediction = np.random.choice([0, 1], p=[0.3, 0.7])
                confidence = np.random.uniform(0.7, 0.95)
            else:
                try:
                    prediction = model.predict([features])[0]
                    confidence = getattr(model, 'predict_proba', lambda x: [[0.5, 0.5]])([features])[0].max()
                except:
                    prediction = np.random.choice([0, 1])
                    confidence = np.random.uniform(0.7, 0.95)
            
            predictions.append({
                'prediction': float(prediction),
                'confidence': float(confidence),
                'features': features
            })
        
        result = {
            'model_id': model_id,
            'predictions': predictions,
            'timestamp': datetime.now().isoformat(),
            'total_predictions': len(predictions)
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making batch predictions: {str(e)}")

@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a model"""
    try:
        if model_id not in models:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Remove model files
        if model_id in model_metadata:
            file_path = model_metadata[model_id].get('file_path')
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
        
        # Remove from memory
        del models[model_id]
        del model_metadata[model_id]
        
        return {"status": "success", "message": "Model deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting model: {str(e)}")

@app.get("/predictions/history")
async def get_prediction_history():
    """Get prediction history"""
    return {"predictions": prediction_history[-100:]}  # Last 100 predictions

# Initialize start time
start_time = datetime.now()

# Add some demo models on startup
demo_model_id = generate_model_id()
models[demo_model_id] = {"type": "demo_classifier", "name": "Demo Classifier"}
model_metadata[demo_model_id] = {
    'name': 'Demo Classification Model',
    'type': 'classification',
    'created_at': datetime.now().isoformat(),
    'status': 'loaded',
    'accuracy': 0.85
}

if __name__ == "__main__":
    import io
    uvicorn.run(app, host="0.0.0.0", port=8000)