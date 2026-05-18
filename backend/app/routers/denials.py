from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def list_denials():
    return {"message": "List denials"}


@router.get("/{denial_id}")
async def get_denial(denial_id: int):
    return {"message": f"Get denial {denial_id}"}
