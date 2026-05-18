from fastapi import APIRouter

router = APIRouter()


@router.post("/predict")
async def predict_denial():
    return {"message": "Denial prediction endpoint"}
