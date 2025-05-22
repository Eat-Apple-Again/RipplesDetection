from fastapi import FastAPI
from fastapi_mcp import FastApiMCP
from app.routers.detect import router as detect_router
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="Ripples Detection API",
    version="0.1.0",
    description="提供水花影像分析：單張 pixel 計算、日期資料夾內的逐日趨勢分析"
)

app.include_router(detect_router)

@app.get("/", summary="Health Check")
async def health():
    return {"status": "ok"}

# MCP
mcp = FastApiMCP(app)
mcp.mount()

@app.get("/mcp/discover", include_in_schema=False, summary="MCP Tools Discover")
async def discover_mcp():
    return {
        "tools": [
            t.dict() if hasattr(t, "dict") else t.model_dump()
            for t in mcp.tools
        ]
    }