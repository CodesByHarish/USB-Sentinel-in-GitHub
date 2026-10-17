import uvicorn
from backend.config import settings

if __name__ == "__main__":
    print(f"Starting Live Voice-Based BRD Interview Agent on http://localhost:{settings.PORT}")
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )
