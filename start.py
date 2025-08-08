#!/usr/bin/env python3
import sys
import subprocess
import os

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def ensure_packages():
    required_packages = [
        "python-dotenv>=1.1.1",
        "fastmcp>=2.11.2", 
        "httpx>=0.24.0",
        "markdownify>=1.1.0",
        "pillow>=11.3.0",
        "pydantic>=2.0.0",
        "readabilipy>=0.3.0",
        "beautifulsoup4>=4.13.4"
    ]
    
    for package in required_packages:
        try:
            print(f"Ensuring {package} is installed...")
            install_package(package)
        except Exception as e:
            print(f"Warning: Could not install {package}: {e}")

if __name__ == "__main__":
    print("Starting MCP server with dependency check...")
    
    # Ensure all packages are installed
    ensure_packages()
    
    # Set environment variables
    os.environ.setdefault("PORT", "10000")
    
    print("Starting main application...")
    
    # Import and run the main application
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "mcp-bearer-token"))
    
    try:
        import mcp_starter
        import asyncio
        asyncio.run(mcp_starter.main())
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)
