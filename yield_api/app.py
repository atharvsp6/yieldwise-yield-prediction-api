# ============================================================================
# YieldWise: Crop Yield Prediction API
# Production-ready FastAPI microservice for India-focused agriculture
# ============================================================================

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# MODEL PATHS
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / 'models'

MODEL_PATH = MODELS_DIR / 'yieldwise_gov_model.joblib'
FEATURES_CONFIG_PATH = MODELS_DIR / 'yieldwise_gov_features.json'

logger.info(f"Base directory: {BASE_DIR}")
logger.info(f"Models directory: {MODELS_DIR}")

# Static files directory
STATIC_DIR = Path(__file__).parent / 'static'
logger.info(f"Static directory: {STATIC_DIR}")

# ============================================================================
# PYDANTIC MODELS (Request/Response Schemas)
# ============================================================================

class CropYieldRequest(BaseModel):
    """
    Input schema for crop yield prediction (Government Data Model v1).
    
    Uses only government of India crop production data features.
    All fields are required and validated.
    """
    
    # Categorical features
    state: str = Field(
        ...,
        example="punjab",
        description="Indian state name (lowercase)"
    )
    district: str = Field(
        ...,
        example="ludhiana",
        description="District name within the state (lowercase)"
    )
    crop: str = Field(
        ...,
        example="rice",
        description="Crop type (e.g., rice, wheat, cotton, sugarcane)"
    )
    season: str = Field(
        ...,
        example="kharif",
        description="Growing season (kharif, rabi, or zaid)"
    )
    
    # Numerical features
    year: int = Field(
        ...,
        example=2014,
        description="Year of cultivation",
        ge=1997,
        le=2024
    )
    area: float = Field(
        ...,
        example=50000.0,
        description="Cultivated area in hectares",
        gt=0
    )
    
    @validator('state', 'district', 'crop', 'season')
    def validate_string_fields(cls, v):
        """Ensure string fields are not empty."""
        if not v or len(v.strip()) == 0:
            raise ValueError('Field cannot be empty')
        return v.strip().lower()
    
    class Config:
        schema_extra = {
            "example": {
                "state": "punjab",
                "district": "ludhiana",
                "crop": "rice",
                "season": "kharif",
                "year": 2014,
                "area": 50000.0
            }
        }


class CropYieldResponse(BaseModel):
    """
    Output schema for crop yield prediction.
    """
    
    predicted_yield: float = Field(
        ...,
        description="Predicted crop yield in kg/ha"
    )
    unit: str = Field(
        default="kg/ha",
        description="Unit of yield measurement"
    )
    input_summary: Dict[str, Any] = Field(
        ...,
        description="Summary of input parameters used"
    )
    model_name: str = Field(
        default="YieldWise Core v1 (Government Data Model)",
        description="Model name and version"
    )
    status: str = Field(
        default="success",
        description="Prediction status"
    )


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_path: str = Field(..., description="Path to loaded model")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_model_and_config() -> tuple:
    """
    Load the trained model and feature configuration.
    
    Returns:
        tuple: (model, feature_config)
    
    Raises:
        FileNotFoundError: If model or config files don't exist
    """
    logger.info("Loading model and configuration...")
    
    # Check if model file exists
    if not MODEL_PATH.exists():
        logger.error(f"Model not found at {MODEL_PATH}")
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
    # Check if features config exists
    if not FEATURES_CONFIG_PATH.exists():
        logger.error(f"Features config not found at {FEATURES_CONFIG_PATH}")
        raise FileNotFoundError(f"Features config not found at {FEATURES_CONFIG_PATH}")
    
    # Load model with protocol compatibility
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=Warning)
        try:
            model = joblib.load(MODEL_PATH)
            # Force set the attribute that's causing issues
            if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
                preprocessor = model.named_steps['preprocessor']
                if hasattr(preprocessor, '_name_to_fitted_passthrough'):
                    pass
                else:
                    # Set the missing attribute
                    preprocessor._name_to_fitted_passthrough = {}
        except Exception as e:
            logger.warning(f"Loading model with standard method, retrying with compatibility mode: {e}")
            model = joblib.load(MODEL_PATH)
    
    logger.info(f"✅ Model loaded from {MODEL_PATH}")
    
    # Load features config
    with open(FEATURES_CONFIG_PATH, 'r') as f:
        config = json.load(f)
    logger.info(f"✅ Features config loaded from {FEATURES_CONFIG_PATH}")
    
    return model, config


def build_input_df(request: CropYieldRequest, feature_order: list) -> pd.DataFrame:
    """
    Convert Pydantic request to DataFrame with correct column order.
    
    Args:
        request: CropYieldRequest object
        feature_order: List of feature names in correct order
    
    Returns:
        pd.DataFrame: Single-row DataFrame ready for prediction
    """
    # Convert request to dictionary
    request_dict = request.dict()

    # Normalize inputs if options available
    if VALID_OPTIONS:
        # Normalize Crop (Handle "Groundnuts" -> "groundnut")
        raw_crop = request_dict['crop'].lower().strip()
        valid_crops = VALID_OPTIONS.get('crops', [])
        
        if raw_crop in valid_crops:
            request_dict['crop'] = raw_crop
        elif raw_crop.endswith('s') and raw_crop[:-1] in valid_crops:
            # Handle pluralization (e.g. groundnuts -> groundnut)
            request_dict['crop'] = raw_crop[:-1]
            logger.info(f"Normalized crop '{request.crop}' to '{request_dict['crop']}'")
        else:
            # Fallback to lower
            request_dict['crop'] = raw_crop
            
        # Normalize Season & District
        request_dict['season'] = request_dict['season'].lower().strip()
        request_dict['district'] = request_dict['district'].lower().strip()
        
        # Normalize State (Case Insensitive Match)
        raw_state = request_dict['state'].strip()
        valid_states = VALID_OPTIONS.get('states', [])
        # Find exact match insensitive
        found_state = next((s for s in valid_states if s.lower() == raw_state.lower()), None)
        if found_state:
            request_dict['state'] = found_state
            if found_state != raw_state:
                 logger.info(f"Normalized state '{raw_state}' to '{found_state}'")
    
    # Create DataFrame from request
    df = pd.DataFrame([request_dict])
    
    # Ensure columns are in correct order
    df = df[feature_order]
    
    logger.debug(f"Input DataFrame shape: {df.shape}")
    logger.debug(f"Input columns: {list(df.columns)}")
    
    return df


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

# Create FastAPI app
app = FastAPI(
    title="🌾 YieldWise API",
    description="Production-ready Crop Yield Prediction API for India",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    logger.info("✅ Static files mounted")
else:
    logger.warning(f"⚠️ Static directory not found: {STATIC_DIR}")

# Global variables (loaded at startup)
MODEL = None
FEATURES_CONFIG = None
OPTIONS_PATH = STATIC_DIR / 'form_options.json'
VALID_OPTIONS = {}

# ============================================================================
# STARTUP & SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model and config at application startup."""
    global MODEL, FEATURES_CONFIG, VALID_OPTIONS
    
    logger.info("="*80)
    logger.info("🌾 YieldWise API Starting Up (Government Data Model v1)...")
    logger.info("="*80)
    
    try:
        MODEL, FEATURES_CONFIG = load_model_and_config()
        logger.info("✅ Model and configuration loaded successfully")
        logger.info(f"   Categorical Features: {FEATURES_CONFIG.get('categorical_features', [])}")
        logger.info(f"   Numerical Features: {FEATURES_CONFIG.get('numerical_features', [])}")
        logger.info(f"   Best Model: {FEATURES_CONFIG.get('best_model', 'Unknown')}")
        logger.info(f"   R² Score: {FEATURES_CONFIG.get('r2', 'N/A'):.4f}")
        logger.info(f"   RMSE: {FEATURES_CONFIG.get('rmse', 'N/A'):.2f} kg/ha")
        
        # Load valid options for normalization
        if OPTIONS_PATH.exists():
            with open(OPTIONS_PATH, 'r') as f:
                VALID_OPTIONS = json.load(f)
            logger.info(f"✅ Loaded valid options: {len(VALID_OPTIONS.get('crops', []))} crops found")
        else:
            logger.warning("⚠️ form_options.json not found - input normalization limited")
            
        logger.info("="*80)
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("🌾 YieldWise API Shutting Down...")
    logger.info("="*80)


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/index.html")
@app.get("/")
async def serve_frontend() -> FileResponse:
    """
    Serve the frontend HTML page.
    
    Example:
        curl http://localhost:8000/
    """
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path, media_type="text/html")
    else:
        raise HTTPException(status_code=404, detail="Frontend not found")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/api/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns:
        HealthResponse: Service status and model information
    
    Example:
        curl http://localhost:8000/api/health
    """
    return HealthResponse(
        status="healthy" if MODEL is not None else "unhealthy",
        model_loaded=MODEL is not None,
        model_path=str(MODEL_PATH)
    )


@app.post("/predict-yield", response_model=CropYieldResponse)
async def predict_yield(request: CropYieldRequest) -> CropYieldResponse:
    """
    Predict crop yield based on input parameters.
    
    Args:
        request: CropYieldRequest with all required fields
    
    Returns:
        CropYieldResponse: Predicted yield and summary
    
    Raises:
        HTTPException: If model is not loaded or prediction fails
    
    Example:
        curl -X POST "http://localhost:8000/predict-yield" \
          -H "Content-Type: application/json" \
          -d '{
            "state": "Punjab",
            "district": "Ludhiana",
            "crop": "rice",
            "season": "kharif",
            "year": 2020,
            "area": 50000.0,
            "rainfall": 800.0,
            "seasonal_rainfall": 750.0,
            "temperature": 28.0,
            "humidity": 75.0,
            "soil_ph": 6.8,
            "fertilizer": 150.0,
            "pesticide": 5.0
          }'
    """
    
    # Check if model is loaded
    if MODEL is None:
        logger.error("Model not loaded")
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service temporarily unavailable."
        )
    
    try:
        logger.info(f"Received prediction request: {request.state}/{request.crop}/{request.season}")
        logger.info(f"Request data: state={request.state}, district={request.district}, crop={request.crop}, season={request.season}")
        
        # Build input DataFrame with correct feature order
        feature_order = FEATURES_CONFIG['all_features']
        input_df = build_input_df(request, feature_order)
        logger.debug(f"Input DataFrame shape: {input_df.shape}, values: {input_df.to_dict('records')}")
        
        # Make prediction with compatibility handling
        logger.debug("Running model prediction...")
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=Warning)
            try:
                prediction = MODEL.predict(input_df)[0]
            except AttributeError as e:
                # Handle version compatibility issues
                if '_name_to_fitted_passthrough' in str(e):
                    logger.warning("Applying sklearn compatibility fix...")
                    # Fix the preprocessor
                    if hasattr(MODEL, 'named_steps') and 'preprocessor' in MODEL.named_steps:
                        MODEL.named_steps['preprocessor']._name_to_fitted_passthrough = {}
                    prediction = MODEL.predict(input_df)[0]
                else:
                    raise
        
        logger.info(f"Raw prediction value: {prediction}")
        
        # Ensure prediction is valid
        if prediction < 0:
            logger.warning(f"Model returned negative prediction: {prediction}. Clamping to 0.")
            prediction = 0.0
        elif prediction == 0.0 or prediction < 100:
            logger.warning(f"⚠️ Unusually low yield prediction: {prediction:.2f} kg/ha - check input data")
        
        logger.info(f"✅ Prediction successful: {prediction:.2f} kg/ha")
        
        # Create response
        response = CropYieldResponse(
            predicted_yield=round(prediction, 2),
            unit="kg/ha",
            input_summary={
                "state": request.state,
                "district": request.district,
                "crop": request.crop,
                "season": request.season,
                "year": request.year,
                "area_ha": request.area
            },
            model_name="YieldWise Core v1 (Government Data Model)",
            status="success"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/model-info")
async def model_info() -> Dict[str, Any]:
    """
    Get information about the loaded model.
    
    Returns:
        Dict: Model metadata and configuration
    
    Example:
        curl http://localhost:8000/model-info
    """
    if FEATURES_CONFIG is None:
        raise HTTPException(
            status_code=503,
            detail="Model configuration not loaded"
        )
    
    return {
        "model_name": "YieldWise Core v1 (Government Data Model)",
        "best_model": FEATURES_CONFIG.get('best_model'),
        "features": {
            "categorical": FEATURES_CONFIG.get('categorical_features', []),
            "numerical": FEATURES_CONFIG.get('numerical_features', [])
        },
        "performance_metrics": {
            "r2_score": FEATURES_CONFIG.get('r2'),
            "rmse_kg_ha": FEATURES_CONFIG.get('rmse'),
            "mae_kg_ha": FEATURES_CONFIG.get('mae')
        },
        "target": FEATURES_CONFIG.get('target'),
        "data_source": "Government of India crop production data (data.gov.in)"
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError exceptions."""
    logger.error(f"ValueError: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content={"detail": f"Invalid input: {str(exc)}"}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# ============================================================================
# MAIN RUNNER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("\n" + "="*80)
    logger.info("🌾 Starting YieldWise API Server")
    logger.info("="*80)
    logger.info("\n📊 API Documentation:")
    logger.info("   Swagger UI:  http://localhost:8000/docs")
    logger.info("   ReDoc:       http://localhost:8000/redoc")
    logger.info("\n📋 Endpoints:")
    logger.info("   GET  /                 - Health check")
    logger.info("   POST /predict-yield    - Make prediction")
    logger.info("   GET  /model-info       - Model information")
    logger.info("\n" + "="*80)
    
    # Run uvicorn server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
