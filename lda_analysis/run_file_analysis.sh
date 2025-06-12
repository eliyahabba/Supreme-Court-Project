#!/bin/bash

# File Length Analysis Runner
# עוזר להפעלת ניתוח אורך קבצים עבור LDA

echo "📝 Supreme Court File Length Analysis Runner"
echo "============================================"

# בדיקת מיקום הסקריפט
if [ ! -f "analyze_file_lengths.py" ]; then
    echo "❌ Error: analyze_file_lengths.py not found"
    echo "Please run this script from the lda_analysis directory"
    exit 1
fi

# הפעלת ניתוח אורך קבצים
echo "🔄 Running file length analysis..."
python analyze_file_lengths.py

# בדיקת הצלחה
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ File length analysis completed successfully!"
    echo ""
    echo "📋 Next steps:"
    echo "1. Run LDA analysis with file filtering:"
    echo "   python generate_analysis_files.py --mode both"
    echo ""
    echo "2. Or run Streamlit app:"
    echo "   streamlit run streamlit_app.py"
    echo ""
    echo "3. To disable filtering, add --no-length-filter flag"
else
    echo ""
    echo "❌ File length analysis failed!"
    echo ""
    echo "🔧 Troubleshooting tips:"
    echo "1. Check that text files are in the correct directory"
    echo "2. Verify the required sample files exist:"
    echo "   - 00003780-A03.txt"
    echo "   - 00003890-I02.txt"
    echo "   - 00003900-G02.txt"
    echo "   - 00003920-A02.txt"
    echo "   - 00003980-G03.txt"
    echo "   - 00003990-V04.txt"
    echo ""
    echo "3. Check file permissions and encoding"
fi 