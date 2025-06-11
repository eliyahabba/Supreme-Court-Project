#!/bin/bash

echo "🚀 מפעיל אפליקציית Streamlit לניתוח LDA..."
echo "=================================="
echo ""
echo "האפליקציה תיפתח בדפדפן בכתובת: http://localhost:8501"
echo "לעצירה: Ctrl+C"
echo ""

# Change to lda_analysis directory
cd "$(dirname "$0")"

# Run Streamlit
streamlit run streamlit_app.py 