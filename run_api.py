#!/usr/bin/env python
"""
Script để chạy ALPR API server
"""
import uvicorn

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 Starting ALPR API Server...")
    print("=" * 50)
    print("📍 API URL: http://127.0.0.1:8000")
    print("📚 Swagger UI: http://127.0.0.1:8000/docs")
    print("🔄 Auto-reload: Enabled")
    print("=" * 50)
    print("\n💡 Press Ctrl+C to stop the server\n")
    
    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,  # Auto-reload khi code thay đổi
        log_level="info"
    )

