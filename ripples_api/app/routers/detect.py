# ripples_api/app/routers/detect.py
from fastapi import APIRouter, Query, HTTPException
from app.services.ripple_detection import current_feed_status, past_feed_status

router = APIRouter(prefix="/detect", tags=["detection"])

@router.get("/current_status")
async def current_status(
    frames_dir: str = Query(...),
    magic_mask: str | None = Query(None)
):
    try:
        res = current_feed_status(frames_dir, resource_dir="app/resource", magic_mask_path=magic_mask)
        return res
    except Exception as e:
        raise HTTPException(400, str(e))

@router.get("/past_status")
async def past_status(
    frames_dir: str = Query(...),
    magic_mask: str | None = Query(None)
):
    try:
        res = past_feed_status(frames_dir, resource_dir="app/resource", magic_mask_path=magic_mask)
        return res
    except Exception as e:
        raise HTTPException(400, str(e))
