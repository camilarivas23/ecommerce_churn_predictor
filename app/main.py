"""
FastAPI Application - E-Commerce Churn Predictor
Serves predictions via REST API and interactive web UI.
"""
import os
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

try:
    from app.model import ChurnPredictor, load_model
except ImportError:
    from model import ChurnPredictor, load_model

# ============================================================
# Config
# ============================================================
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), "model.pkl"))
FEATURES_PATH = os.environ.get("FEATURES_PATH", os.path.join(os.path.dirname(__file__), "features.json"))

app = FastAPI(
    title="E-Commerce Churn Predictor",
    description="Predicts if a user will stop purchasing based on cart behavior",
    version="1.0.0",
)

predictor: Optional[ChurnPredictor] = None


# ============================================================
# Pydantic Models
# ============================================================
class ChurnPredictRequest(BaseModel):
    days_since_last_cart: int = Field(..., description="Days since last cart activity", ge=0)
    is_active_last_7d: int = Field(..., description="Active in last 7 days", ge=0, le=1)
    is_active_last_30d: int = Field(..., description="Active in last 30 days", ge=0, le=1)
    total_orders: int = Field(..., description="Total number of orders")
    total_items_purchased: int = Field(..., description="Total items purchased")
    unique_products_bought: int = Field(..., description="Number of unique products bought")
    avg_cart_size: float = Field(..., description="Average items per cart")
    unique_categories_bought: int = Field(..., description="Number of unique categories bought")
    total_spent: float = Field(..., description="Total amount spent")
    avg_order_value: float = Field(..., description="Average order value")
    avg_product_rating: float = Field(..., description="Average rating of bought products")
    avg_discount_pct: float = Field(..., description="Average discount percentage used")
    avg_price_per_item: float = Field(..., description="Average price per item")


class TopFeature(BaseModel):
    name: str
    importance: float
    value: float


class ChurnResponse(BaseModel):
    user_id: str
    prediction: str
    churn_probability: float
    confidence: float
    top_features: List[TopFeature]
    timestamp: str
    interpretation: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    prediction_type: str


# ============================================================
# Startup
# ============================================================
@app.on_event("startup")
async def startup_event():
    global predictor
    try:
        predictor = load_model(MODEL_PATH, FEATURES_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"WARNING: Could not load model: {e}")
        predictor = None


# ============================================================
# Endpoints
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
    <head><title>E-Commerce Churn Predictor API</title></head>
    <body style="font-family: sans-serif; max-width: 600px; margin: 40px auto; padding: 0 20px;">
        <h1>E-Commerce Churn Predictor</h1>
        <p>Predicts if a user will stop purchasing based on cart behavior.</p>
        <ul>
            <li><a href="/docs">Swagger UI</a> - Interactive API docs</li>
            <li><a href="/predict">Web UI</a> - Prediction form</li>
            <li><a href="/health">Health Check</a></li>
        </ul>
    </body>
    </html>
    """


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        timestamp=datetime.now().isoformat(),
        model_loaded=predictor is not None,
        prediction_type="e-commerce_user_churn",
    )


@app.get("/model-info")
async def model_info():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return predictor.get_model_info()


@app.get("/predict", response_class=HTMLResponse)
async def predict_form():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>E-Commerce Churn Predictor</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; padding: 2rem; }
            .container { max-width: 700px; margin: 0 auto; }
            h1 { color: #38bdf8; margin-bottom: 0.5rem; font-size: 1.8rem; }
            .subtitle { color: #94a3b8; margin-bottom: 2rem; }
            .form-group { margin-bottom: 1rem; }
            label { display: block; color: #94a3b8; margin-bottom: 0.3rem; font-size: 0.9rem; }
            input { width: 100%; padding: 0.6rem; border: 1px solid #334155; border-radius: 6px; background: #1e293b; color: #e2e8f0; font-size: 1rem; }
            input:focus { outline: none; border-color: #38bdf8; }
            input:read-only { background: #0f172a; color: #38bdf8; border-color: #1e3a5f; cursor: default; }
            .row { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
            .section-label { color: #38bdf8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; margin: 1.5rem 0 0.5rem; border-bottom: 1px solid #1e3a5f; padding-bottom: 0.3rem; }
            button { width: 100%; padding: 0.8rem; background: #2563eb; color: white; border: none; border-radius: 6px; font-size: 1.1rem; cursor: pointer; margin-top: 1rem; }
            button:hover { background: #1d4ed8; }
            .result { margin-top: 2rem; padding: 1.5rem; border-radius: 8px; display: none; }
            .result.active { display: block; }
            .result.churned { background: #450a0a; border: 1px solid #dc2626; }
            .result.active-user { background: #052e16; border: 1px solid #16a34a; }
            .result h2 { margin-bottom: 0.5rem; }
            .bar-container { margin: 0.5rem 0; }
            .bar-label { font-size: 0.85rem; color: #94a3b8; }
            .bar-bg { background: #334155; border-radius: 4px; height: 8px; overflow: hidden; }
            .bar-fill { height: 100%; border-radius: 4px; transition: width 0.5s; }
            .features { margin-top: 1rem; }
            .feature-item { display: flex; justify-content: space-between; padding: 0.3rem 0; border-bottom: 1px solid #334155; font-size: 0.9rem; }
            .hint { font-size: 0.75rem; color: #64748b; margin-top: 0.2rem; }
            .error { color: #f87171; font-size: 0.8rem; margin-top: 0.2rem; display: none; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>E-Commerce Churn Predictor</h1>
            <p class="subtitle">Predict if a user will stop purchasing based on cart behavior</p>

            <form id="predictForm">
                <div class="section-label">Recency (Activity)</div>
                <div class="row">
                    <div class="form-group">
                        <label>Days Since Last Cart</label>
                        <input type="number" id="days_since_last_cart" value="5" min="0" step="1" required>
                        <div class="hint">0 = active today</div>
                    </div>
                    <div class="form-group">
                        <label>Active Last 7 Days</label>
                        <input type="number" id="is_active_last_7d" readonly>
                        <div class="hint">= 1 if days_since_last_cart <= 7</div>
                    </div>
                </div>
                <div class="form-group">
                    <label>Active Last 30 Days</label>
                    <input type="number" id="is_active_last_30d" readonly>
                    <div class="hint">= 1 if days_since_last_cart <= 30</div>
                </div>

                <div class="section-label">Order Behavior</div>
                <div class="row">
                    <div class="form-group">
                        <label>Total Orders</label>
                        <input type="number" id="total_orders" value="5" min="1" step="1" required>
                        <div class="hint">Minimum 1 order</div>
                    </div>
                    <div class="form-group">
                        <label>Total Items Purchased</label>
                        <input type="number" id="total_items_purchased" value="15" min="1" step="1" required>
                        <div class="hint">Must be >= total orders</div>
                    </div>
                </div>

                <div class="section-label">Product Diversity</div>
                <div class="row">
                    <div class="form-group">
                        <label>Unique Products Bought</label>
                        <input type="number" id="unique_products_bought" value="8" min="1" step="1" required>
                        <div class="hint">Must be <= total items</div>
                    </div>
                    <div class="form-group">
                        <label>Unique Categories</label>
                        <input type="number" id="unique_categories_bought" value="3" min="1" step="1" required>
                        <div class="hint">Must be <= unique products</div>
                    </div>
                </div>

                <div class="section-label">Spending</div>
                <div class="row">
                    <div class="form-group">
                        <label>Total Spent ($)</label>
                        <input type="number" id="total_spent" value="500.00" min="0.01" step="0.01" required>
                        <div class="hint">Minimum $0.01</div>
                    </div>
                    <div class="form-group">
                        <label>Avg Product Rating</label>
                        <input type="number" id="avg_product_rating" value="4.2" min="1" max="5" step="0.1" required>
                        <div class="hint">Range: 1.0 - 5.0</div>
                    </div>
                </div>

                <div class="section-label">Discounts</div>
                <div class="form-group">
                    <label>Avg Discount %</label>
                    <input type="number" id="avg_discount_pct" value="12.5" min="0" max="100" step="0.1" required>
                    <div class="hint">Range: 0% - 100%</div>
                </div>

                <div class="section-label">Calculated Averages (auto-filled)</div>
                <div class="row">
                    <div class="form-group">
                        <label>Avg Cart Size</label>
                        <input type="number" id="avg_cart_size" readonly>
                        <div class="hint">= total items / total orders</div>
                    </div>
                    <div class="form-group">
                        <label>Avg Order Value ($)</label>
                        <input type="number" id="avg_order_value" readonly>
                        <div class="hint">= total spent / total orders</div>
                    </div>
                </div>
                <div class="form-group">
                    <label>Avg Price Per Item ($)</label>
                    <input type="number" id="avg_price_per_item" readonly>
                    <div class="hint">= total spent / total items</div>
                </div>

                <button type="submit">Predict Churn</button>
            </form>

            <div id="result" class="result"></div>
        </div>

        <script>
        const fields = ['days_since_last_cart','total_orders','total_items_purchased','unique_products_bought','unique_categories_bought','total_spent','avg_product_rating','avg_discount_pct'];
        const calcFields = ['is_active_last_7d','is_active_last_30d','avg_cart_size','avg_order_value','avg_price_per_item'];

        function recalc() {
            const v = {};
            fields.forEach(f => { v[f] = parseFloat(document.getElementById(f).value) || 0; });

            // Derive recency flags from days_since_last_cart
            document.getElementById('is_active_last_7d').value = v.days_since_last_cart <= 7 ? 1 : 0;
            document.getElementById('is_active_last_30d').value = v.days_since_last_cart <= 30 ? 1 : 0;

            if (v.total_orders > 0 && v.total_items_purchased > 0) {
                document.getElementById('avg_cart_size').value = (v.total_items_purchased / v.total_orders).toFixed(2);
                document.getElementById('avg_order_value').value = (v.total_spent / v.total_orders).toFixed(2);
                document.getElementById('avg_price_per_item').value = (v.total_spent / v.total_items_purchased).toFixed(2);
            }
        }

        fields.forEach(f => {
            document.getElementById(f).addEventListener('input', recalc);
        });

        recalc();

        document.getElementById('predictForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const data = {};
            calcFields.concat(fields).forEach(f => {
                data[f] = parseFloat(document.getElementById(f).value) || 0;
            });
            try {
                const resp = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data),
                });
                const result = await resp.json();
                const div = document.getElementById('result');
                div.className = 'result active ' + (result.prediction === 'CHURNED' ? 'churned' : 'active-user');
                div.innerHTML = `
                    <h2>${result.prediction === 'CHURNED' ? 'Churn Risk' : 'Active User'}</h2>
                    <div class="bar-container">
                        <div class="bar-label">Churn Probability: ${(result.churn_probability * 100).toFixed(1)}%</div>
                        <div class="bar-bg"><div class="bar-fill" style="width:${result.churn_probability * 100}%;background:${result.prediction === 'CHURNED' ? '#dc2626' : '#16a34a'}"></div></div>
                    </div>
                    <p style="margin-top:0.5rem;color:#94a3b8;font-size:0.9rem;">Confidence: ${(result.confidence * 100).toFixed(1)}%</p>
                    <p style="margin-top:0.5rem;font-size:0.95rem;">${result.interpretation}</p>
                    <div class="features">
                        <strong>Top Predictive Features:</strong>
                        ${result.top_features.map(f => `<div class="feature-item"><span>${f.name}</span><span>${f.value} (imp: ${(f.importance * 100).toFixed(1)}%)</span></div>`).join('')}
                    </div>
                `;
            } catch (err) {
                alert('Error: ' + err.message);
            }
        });
        </script>
    </body>
    </html>
    """


@app.post("/predict", response_model=ChurnResponse)
async def predict(request: ChurnPredictRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    f = request.model_dump()

    # Server-side validation: enforce realistic values
    if f["days_since_last_cart"] < 0:
        raise HTTPException(status_code=400, detail="days_since_last_cart must be >= 0")
    if f["total_orders"] < 1:
        raise HTTPException(status_code=400, detail="total_orders must be >= 1")
    if f["total_items_purchased"] < f["total_orders"]:
        raise HTTPException(status_code=400, detail="total_items_purchased must be >= total_orders")
    if f["unique_products_bought"] < 1 or f["unique_products_bought"] > f["total_items_purchased"]:
        raise HTTPException(status_code=400, detail="unique_products_bought must be between 1 and total_items_purchased")
    if f["unique_categories_bought"] < 1 or f["unique_categories_bought"] > f["unique_products_bought"]:
        raise HTTPException(status_code=400, detail="unique_categories_bought must be between 1 and unique_products_bought")
    if f["total_spent"] <= 0:
        raise HTTPException(status_code=400, detail="total_spent must be > 0")
    if f["avg_product_rating"] < 1 or f["avg_product_rating"] > 5:
        raise HTTPException(status_code=400, detail="avg_product_rating must be between 1 and 5")
    if f["avg_discount_pct"] < 0 or f["avg_discount_pct"] > 100:
        raise HTTPException(status_code=400, detail="avg_discount_pct must be between 0 and 100")

    # Auto-calculate derived fields
    f["avg_cart_size"] = round(f["total_items_purchased"] / f["total_orders"], 2)
    f["avg_order_value"] = round(f["total_spent"] / f["total_orders"], 2)
    f["avg_price_per_item"] = round(f["total_spent"] / f["total_items_purchased"], 2)

    result = predictor.predict(f)

    interpretation = _generate_interpretation(
        result["prediction"], result["churn_probability"], result["top_features"]
    )

    return ChurnResponse(
        user_id="user_input",
        prediction=result["prediction"],
        churn_probability=result["churn_probability"],
        confidence=result["confidence"],
        top_features=[TopFeature(**f) for f in result["top_features"]],
        timestamp=datetime.now().isoformat(),
        interpretation=interpretation,
    )


def _generate_interpretation(prediction, probability, top_features):
    """Generate human-readable interpretation."""
    if prediction == "CHURNED":
        if probability > 0.8:
            level = "HIGH"
        elif probability > 0.6:
            level = "MODERATE"
        else:
            level = "LOW"
        return (
            f"This user has a {level} churn risk ({probability:.1%}). "
            f"Top factors: {top_features[0]['name']}={top_features[0]['value']}. "
            f"Consider sending a re-engagement discount or personalized recommendations."
        )
    else:
        return (
            f"This user is likely ACTIVE ({(1-probability):.1%} retention probability). "
            f"Keep them engaged with loyalty rewards and new product notifications."
        )
