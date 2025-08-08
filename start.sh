#!/bin/bash
# Render deployment script

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Starting MCP server..."
cd mcp-bearer-token
python mcp_starter.py
