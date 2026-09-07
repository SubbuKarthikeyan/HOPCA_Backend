import uvicorn
from app.core.config import settings

if __name__ == "__main__":
    print(f"Starting Hospital Operations Agent backend in '{settings.app_env}' mode...")
    print("API Documentation: http://127.0.0.1:8000/docs")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
