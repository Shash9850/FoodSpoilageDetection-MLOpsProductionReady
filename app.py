import os
import shutil

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from src.pipeline.prediction_pipeline import PredictionPipeline

app = FastAPI(title="Food Spoilage Detection API")

# -----------------------------
# Static & Templates
# -----------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# -----------------------------
# Model & Upload Directory
# -----------------------------
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MODEL_PATH = "artifacts/models/food_spoilage_cnn.pth"

CLASS_NAMES = [
    "freshapples",
    "freshbanana",
    "freshbittergroud",
    "freshcapsicum",
    "freshcucumber",
    "freshokra",
    "freshoranges",
    "freshpotato",
    "freshtomato",
    "rottenapples",
    "rottenbanana",
    "rottenbittergroud",
    "rottencapsicum",
    "rottencucumber",
    "rottenokra",
    "rottenoranges",
    "rottenpotato",
    "rottentomato"
]

predictor = PredictionPipeline(
    model_path=MODEL_PATH,
    class_names=CLASS_NAMES
)


# -----------------------------
# API Routes
# -----------------------------
@app.get("/")
def health_check():
    return {"status": "running"}

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction = predictor.predict(file_path)

    return {
        "filename": file.filename,
        "prediction": prediction
    }

# -----------------------------
# UI Routes
# -----------------------------
@app.get("/ui", response_class=HTMLResponse)
def ui_home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.post("/ui-predict", response_class=HTMLResponse)
async def ui_predict(request: Request, file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction = predictor.predict(file_path)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": prediction
        }
    )
