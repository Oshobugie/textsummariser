from fastapi import FastAPI
import uvicorn
import sys
import os
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from pydantic import BaseModel # <--- ADD THIS IMPORT
from src.textSummariser.pipeline.prediction_pipeline import PredictionPipeline

# Define what the body should look like
class TextInput(BaseModel):
    text: str

app = FastAPI()

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def training():
    try:
        os.system("python main.py")
        return Response("Training successful !!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")
    
# Update the route to use the BaseModel
@app.post("/predict")
async def predict_route(input_data: TextInput): # <--- CHANGE THIS LINE
    try:
        obj = PredictionPipeline()
        # Access the text from the body
        text = obj.predict(input_data.text) # <--- CHANGE THIS LINE
        return text
    except Exception as e:
        raise e
    
if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)