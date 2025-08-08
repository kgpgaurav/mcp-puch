#!/usr/bin/env python3

print("Testing imports...")

try:
    import asyncio
    print("✓ asyncio")
except ImportError as e:
    print(f"✗ asyncio: {e}")

try:
    from typing import Annotated
    print("✓ typing")
except ImportError as e:
    print(f"✗ typing: {e}")

try:
    import os
    print("✓ os")
except ImportError as e:
    print(f"✗ os: {e}")

try:
    from dotenv import load_dotenv
    print("✓ python-dotenv")
except ImportError as e:
    print(f"✗ python-dotenv: {e}")

try:
    from fastmcp import FastMCP
    print("✓ fastmcp")
except ImportError as e:
    print(f"✗ fastmcp: {e}")

try:
    from pydantic import BaseModel, Field, AnyUrl
    print("✓ pydantic")
except ImportError as e:
    print(f"✗ pydantic: {e}")

try:
    import markdownify
    print("✓ markdownify")
except ImportError as e:
    print(f"✗ markdownify: {e}")

try:
    import httpx
    print("✓ httpx")
except ImportError as e:
    print(f"✗ httpx: {e}")

try:
    import readabilipy
    print("✓ readabilipy")
except ImportError as e:
    print(f"✗ readabilipy: {e}")

try:
    from PIL import Image
    print("✓ pillow")
except ImportError as e:
    print(f"✗ pillow: {e}")

try:
    from bs4 import BeautifulSoup
    print("✓ beautifulsoup4")
except ImportError as e:
    print(f"✗ beautifulsoup4: {e}")

print("Import test complete!")
