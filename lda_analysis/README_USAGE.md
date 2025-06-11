# מדריך השימוש - ניתוח LDA חדש

## 📁 מיקום הקבצים
כל הקבצים נמצאים בתיקיה: `lda_analysis/`

## קבצים במערכת החדשה

### 1. `lda_visualizations.py`
פונקציות הצגה משותפות לכל הגרפים. כולל:
- פונקציות ליצירת צבעים
- גרפי מגמות יחסיות ומוחלטות
- היסטוגרמות
- התפלגות מוערמת
- ענני מילים

### 2. `streamlit_app.py` 
אפליקציית Streamlit אינטראקטיבית (באנגלית):
- בחירה בין נושא יחיד לרב נושאי
- הגדרת טווח שנים
- הצגה של כל הגרפים בטאבים
- הורדת נתונים

### 3. `generate_analysis_files.py`
סקריפט ליצירת קבצי תוצאות:
- יוצר קבצי HTML
- יוצר קבצי CSV
- יוצר תמונות
- שומר הכל בתיקיות מסודרות

## איך להשתמש

### הפעלת Streamlit (מומלץ לבדיקה מהירה)

**דרך 1: סקריפט מהיר**
```bash
./lda_analysis/run_streamlit.sh
```

**דרך 2: ידנית**
```bash
cd lda_analysis
streamlit run streamlit_app.py
```

האפליקציה תיפתח בדפדפן בכתובת: `http://localhost:8501`

**תכונות האפליקציה:**
- 🎛️ בחירת מצב ניתוח (Single Topic / Multi Topic)
- 📅 בחירת שנה מינימלית לניתוח
- 🎯 הגדרת סף הסתברות (למצב רב נושאי)
- 📊 5 טאבים עם גרפים שונים
- 💾 הורדת נתונים כקבצי CSV
- ⚡ טעינה מהירה עם cache
- 🌐 ממשק באנגלית לנוחות השימוש

### יצירת קבצי תוצאות
```bash
cd lda_analysis

# שני המצבים
python generate_analysis_files.py --mode both

# רק נושא יחיד
python generate_analysis_files.py --mode single

# רק רב נושאי
python generate_analysis_files.py --mode multi

# עם שנה מינימלית
python generate_analysis_files.py --mode both --min-year 1990
```

## תיקיות התוצאות

הקבצים נשמרים ב:
- `data/results/lda/single_topic/` - ניתוח נושא יחיד
- `data/results/lda/multi_topic/` - ניתוח רב נושאי

כל תיקיה מכילה:
- `topics_trend.html` - מגמות יחסיות (אחוזים)
- `absolute_trends.html` - מגמות מוחלטות (מספר מסמכים)
- `stacked_distribution.html` - התפלגות מוערמת לפי שנה
- `topics_histogram.html` - התפלגות כללית של נושאים
- `topics_wordcloud.png` - ענני מילים לכל נושא
- `comprehensive_analysis.html` - דף מאוחד עם כל הגרפים
- `comprehensive_topic_data.csv` - נתונים מפורטים עם תיאורי נושאים
- `yearly_topic_aggregation.csv` - סיכום לפי שנה ונושא
- `topic_mappings_reference.csv` - מפתח נושאים בעברית

## מה השתנה

✅ **פיצול קוד** - פונקציות משותפות בקובץ נפרד
✅ **Streamlit** - צפייה אינטראקטיבית אונליין
✅ **קבצים נפרדים** - גנרטור קבצים נפרד לארכיון
✅ **ממשק נקי** - Streamlit באנגלית, גרפים עם תוויות עבריות
✅ **טעינה מהירה** - cache ב-Streamlit
✅ **בחירות גמישות** - מצבים שונים ושנים

## דרישות

עליך להתקין:
```bash
pip install streamlit
```

שאר הספריות כבר קיימות במערכת.

## טיפים נוספים

### עצירת Streamlit
לעצירת השרת: `Ctrl+C` בטרמינל

### גישה לקבצים שנוצרו
הקבצים נוצרים בתיקיות:
- `data/results/lda/single_topic/` 
- `data/results/lda/multi_topic/`

ניתן לפתוח את קבצי ה-HTML ישירות בדפדפן.

### ההבדל בין המצבים
- **נושא יחיד**: כל מסמך משויך לנושא החזק ביותר בלבד
- **רב נושאי**: מסמכים יכולים להיות מוקצים למספר נושאים (מעל הסף) 