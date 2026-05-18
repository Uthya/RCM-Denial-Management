from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def list_appeals():
    return {"message": "List appeals"}


@router.post("/")
async def create_appeal():
    return {"message": "Create appeal"}
