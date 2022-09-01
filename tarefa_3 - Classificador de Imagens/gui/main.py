import io
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import uvicorn


from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import StreamingResponse, RedirectResponse, JSONResponse
from skimage.io import imread
from skimage.transform import resize

from classifier import RGB2GrayTransformer, HogTransformer


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory=Path(__file__).parent /"templates")


@app.get("/image/{img_id}")
async def get_image(request: Request, img_id: str):

    # open file
    with open(Path(__file__).parent.parent / 'dataset' / 'to_classify' / f'{img_id}', 'rb') as f:
        img = f.read()
        return StreamingResponse(BytesIO(img), media_type="image/jpg")


@app.post("/verification", response_class=HTMLResponse)
async def set_verification(request: Request):
    form_data = await request.form()
    images_ids = form_data.getlist("image[]")
    imgs_type = form_data.get("type")

    dataset_folder = Path(__file__).parent.parent / 'dataset'

    for image_id in images_ids:

        # move file
        (dataset_folder / 'to_classify' / f'{image_id}').rename(
            dataset_folder / imgs_type / f'{image_id}')

    return RedirectResponse(url=request.url, status_code=302)


@app.get("/verification", response_class=HTMLResponse)
async def verification(request: Request):

    all_files = (Path(__file__).parent.parent / 'dataset' / 'to_classify').glob('*.jpg')

    img_names = [x.name for x in all_files]

    return templates.TemplateResponse("verification.html", {"request": request, "images": img_names})


@app.get("/classificar", response_class=HTMLResponse)
async def classificar(request: Request):

    return templates.TemplateResponse("classification.html", {"request": request})


@app.post("/classificar", response_class=JSONResponse)
async def post_classificar(request: Request, file: UploadFile = File(...)):
    form_data = await request.form()

    contents = await file.read()
    test_data = io.BytesIO(contents)

    im = imread(test_data)
    im = resize(im, (80, 80))  # Vamos usar 80 aqui pq é o tamanho que o modelo foi treinado

    hog_pipeline = joblib.load(Path(__file__).parent.parent / 'modelo_treinado' / 'hog_pipeline.pkl')

    prediction = hog_pipeline.predict([im])[0]

    return {"prediction": prediction}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
