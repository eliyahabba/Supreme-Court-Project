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

# שני המצבים (עם סינון אורך קבצים - ברירת מחדל)
python generate_analysis_files.py --mode both

# רק נושא יחיד
python generate_analysis_files.py --mode single

# רק רב נושאי
python generate_analysis_files.py --mode multi

# עם שנה מינימלית
python generate_analysis_files.py --mode both --min-year 1990

# ללא סינון אורך קבצים (כלול את כל הקבצים)
python generate_analysis_files.py --mode both --no-length-filter
```

### 📏 סינון אורך קבצים (תכונה חדשה!)

לפני הפעלת הניתוח, מומלץ לבצע ניתוח אורך הקבצים:

```bash
# בדרך הקלה
./run_file_analysis.sh

# או ידנית
python analyze_file_lengths.py
```

זה ייצור סף אוטומטי לסינון קבצים קצרים מדי. ראה `README_FILE_LENGTH_FILTERING.md` לפרטים מלאים.

### 📝 שימוש בקובץ הגדול (notebooks/lda_analysis.py)

הקובץ הגדול תומך גם בסינון אורך קבצים:

```bash
# עם סינון אורך (ברירת מחדל)
cd notebooks
python lda_analysis.py --mode both

# ללא סינון אורך
python lda_analysis.py --mode both --no-length-filter

# רק ניתוח נושא יחיד עם סינון
python lda_analysis.py --mode single
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

## פורמט יצוא הנתונים

When you download data from either the Streamlit app or generate files with the script, you'll get clean CSV files with these columns:

### Single Topic Mode Columns / עמודות במצב נושא יחיד:
- `filename` - Document filename / שם קובץ המסמך
- `year` - Document year / שנת המסמך  
- `strongest_topic` - Topic ID number / מספר זהוי הנושא
- `strongest_topic_prob` - Confidence score (0-1) / ציון ביטחון
- `topic_title` - Hebrew topic title / כותרת הנושא בעברית
- `topic_description` - Topic keywords and weights / מילות מפתח ומשקלים

### Multi Topic Mode Columns / עמודות במצב נושאים מרובים:
- `filename` - Document filename / שם קובץ המסמך
- `year` - Document year / שנת המסמך
- `strongest_topic` - Topic ID number / מספר זהוי הנושא  
- `strongest_topic_prob` - Topic probability / הסתברות הנושא
- `is_strongest` - Whether this is the strongest topic / האם זה הנושא החזק ביותר
- `topic_title` - Hebrew topic title / כותרת הנושא בעברית
- `topic_description` - Topic keywords and weights / מילות מפתח ומשקלים

**Example CSV format / דוגמה לפורמט CSV:**
```csv
filename,year,strongest_topic,strongest_topic_prob,topic_title,topic_description
21087770-A03.txt,2022,11,0.5849277,הליכים משפטיים,"0.012*""סף"" + 0.011*""הגשה"" + 0.011*""תיק""..."
```

## תיקיות פלט

### Streamlit App / יישום Streamlit
- Data downloads directly to browser / הורדת נתונים ישירות לדפדפן
- Filenames include analysis mode / שמות קבצים כוללים מצב ניתוח

### Static File Generation / יצירת קבצים סטטיים
```
data/results/lda/
├── single_topic/                    # Single topic analysis / ניתוח נושא יחיד
│   ├── comprehensive_topic_data.csv # Clean data export / יצוא נתונים נקי
│   ├── topics_trend.html           # Relative trends / מגמות יחסיות
│   ├── absolute_trends.html        # Absolute trends / מגמות מוחלטות
│   ├── stacked_distribution.html   # Stacked charts / גרפים מוערמים
│   ├── topics_histogram.html       # Topic histogram / היסטוגרמת נושאים
│   ├── topics_wordcloud.png        # Word clouds / ענני מילים
│   ├── comprehensive_analysis.html # Combined dashboard / דשבורד מכולל
│   └── yearly_topic_aggregation.csv # Aggregated data / נתונים מצטברים
└── multi_topic/                     # Multi topic analysis / ניתוח נושאים מרובים
    └── [same structure as single_topic] / [אותה מבנה כמו single_topic]
```

## סוגי גרפים

1. **Relative Trends / מגמות יחסיות** - Shows topic percentages over time / מציג אחוזי נושאים לאורך זמן
2. **Absolute Trends / מגמות מוחלטות** - Shows document counts per topic / מציג מספר מסמכים לכל נושא  
3. **Stacked Distribution / התפלגות מוערמת** - Shows yearly topic composition / מציג הרכב נושאים שנתי
4. **Topic Histogram / היסטוגרמת נושאים** - Shows overall topic frequency / מציג תדירות נושאים כללית
5. **Word Clouds / ענני מילים** - Visual representation of topic keywords / ייצוג ויזואלי של מילות מפתח

## טיפים

### For Streamlit App / ליישום Streamlit:
- Use sidebar controls to filter data / השתמש בבקרות הצד לסינון נתונים
- Download processed data for further analysis / הורד נתונים מעובדים לניתוח נוסף
- Try different analysis modes to compare results / נסה מצבי ניתוח שונים להשוואת תוצאות

### For File Generation / ליצירת קבצים:
- Use `--mode both` to generate comprehensive comparison / השתמש ב-`--mode both` לייצור השוואה מקיפה
- Adjust `--min-year` to focus on specific time periods / התאם `--min-year` להתמקדות בתקופות זמן ספציפיות
- Check the comprehensive_analysis.html for best overview / בדוק את comprehensive_analysis.html לסקירה הטובה ביותר

## פתרון בעיות

### Common Issues / בעיות נפוצות:

1. **"No data found" error / שגיאת "לא נמצאו נתונים"**
   - Check that all required files exist / בדוק שכל הקבצים הנדרשים קיימים
   - Verify file paths in error messages / אמת נתיבי קבצים בהודעות שגיאה

2. **Hebrew text not displaying / טקסט עברי לא מוצג**
   - Install Hebrew fonts on your system / התקן גופנים עבריים במערכת
   - Use a browser that supports Hebrew / השתמש בדפדפן התומך בעברית

3. **Streamlit won't start / Streamlit לא מתחיל**
   - Check Python environment / בדוק סביבת Python
   - Install missing dependencies / התקן תלותות חסרות
   - Try running from project root / נסה להריץ מתיקיית הפרויקט הראשית

4. **Charts not loading / גרפים לא נטענים**
   - Check browser console for errors / בדוק קונסולת הדפדפן לשגיאות
   - Clear browser cache / נקה מטמון הדפדפן
   - Try different browser / נסה דפדפן אחר

### קבלת עזרה:

- Check error messages carefully / בדוק הודעות שגיאה בקפידה
- Ensure all data files are in correct locations / וודא שכל קבצי הנתונים במיקומים הנכונים
- Verify Python environment has required packages / אמת שסביבת Python כוללת חבילות נדרשות 