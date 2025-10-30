#!/bin/bash

echo "🚀 Starting Amazon Analytics App..."
echo ""

# Kill any existing Streamlit processes on port 8501
echo "🔧 Clearing port 8501..."
lsof -ti:8501 | xargs kill -9 2>/dev/null || true

# Wait a moment for cleanup
sleep 2

# Start Streamlit with clean port
echo "✅ Starting Streamlit app..."
echo "📱 App will be available at: http://localhost:8501"
echo ""
echo "ℹ️  Note: Config warnings are harmless and can be ignored"
echo "   (They're from deprecated Streamlit options that were removed)"
echo ""

# Start the app
streamlit run streamlit_app.py --server.port 8501