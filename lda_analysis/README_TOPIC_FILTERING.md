# LDA Analysis with Topic Filtering

## התיקונים החדשים / New Features

### 1. קובץ קבועים משותף / Shared Constants File

נוצר קובץ `constants.py` שמכיל את כל ההגדרות המשותפות:

- **נושאים מוחרגים כברירת מחדל**: נושא 11 (ניתן לשינוי)
- **נתיבי קבצים**: הגדרות נתיבים משותפות לכל המודולים
- **פרמטרים ברירת מחדל**: עבור ניתוח וויזואליזציות

### 2. סינון נושאים וטופ-K / Topic Filtering & Top-K

כל הכלים תומכים כעת בסינון נושאים ותכונות מתקדמות עבור multi-topic:

#### **טופ-K עבור Multi-Topic (חדש!)**
במצב multi-topic, כל מסמך יכול לכלול נושאים על פי שני קריטריונים:
1. **סף הסתברות**: נושאים עם הסתברות מעל הסף (ברירת מחדל: 0.1)
2. **טופ-K**: K הנושאים החזקים ביותר (ברירת מחדל: 3)

**הלוגיקה**: המערכת תיקח את מקסימום בין שני הגישות - אם יש 5 נושאים מעל הסף אבל K=3, היא תיקח את 5. אם יש רק 1 נושא מעל הסף אבל K=3, היא תיקח את 3 החזקים.

#### **סינון נושאים בשתי דרכים**:

#### א. החרגת נושאים / Excluding Topics
```bash
# החרגה של נושא 11 (ברירת מחדל)
python generate_analysis_files.py

# החרגה של נושאים מרובים
python generate_analysis_files.py --exclude-topics 11 15 20
```

#### ב. הכללה של נושאים בלבד / Including Only Specific Topics
```bash
# ניתוח רק של נושאים 1, 5, 10
python generate_analysis_files.py --include-topics 1 5 10
```

### 3. כלים מעודכנים / Updated Tools

#### א. `generate_analysis_files.py`
```bash
# דוגמאות שימוש / Usage Examples:

# ניתוח שני המצבים (single + multi) ללא נושא 11
python generate_analysis_files.py --mode both --exclude-topics 11

# ניתוח multi-topic עם סף גבוה יותר, ללא נושאים 11, 15
python generate_analysis_files.py --mode multi --threshold 0.4 --exclude-topics 11 15

# ניתוח multi-topic עם טופ-5 נושאים חזקים ביותר לכל מסמך
python generate_analysis_files.py --mode multi --top-k 5 --threshold 0.1

# ניתוח רק עם נושאים ספציפיים
python generate_analysis_files.py --include-topics 1 2 3 4 5

# ניתוח מתחיל משנת 2000
python generate_analysis_files.py --min-year 2000
```

**פרמטרים זמינים**:
- `--mode`: single/multi/both (ברירת מחדל: both)
- `--exclude-topics`: רשימת נושאים להחרגה (ברירת מחדל: [11])
- `--include-topics`: רשימת נושאים לכלול (אם מצוין, רק אלה יותרו)
- `--min-year`: שנה מינימלית (ברירת מחדל: 1948)
- `--threshold`: סף הסתברות עבור multi-topic (ברירת מחדל: 0.1)
- `--top-k`: מספר נושאים מובילים לכל מסמך (ברירת מחדל: 3)

#### ב. `streamlit_app.py`
האפליקציה האינטראקטיבית כוללת עכשיו:

- **ממשק סינון נושאים**: בחירה גרפית של נושאים להחרגה או הכללה
- **בקרות Multi-Topic מתקדמות**: סף הסתברות + טופ-K נושאים
- **תצוגת מידע סינון**: הצגת הנושאים שנסננו ומידע על איך נבחרו הנושאים
- **עדכון אוטומטי**: כל הגרפים מתעדכנים לפי הסינון

```bash
# הרצת האפליקציה
streamlit run streamlit_app.py
```

#### ג. `lda_visualizations.py`
כל פונקציות הוויזואליזציה תומכות בסינון:

```python
from lda_visualizations import plot_topics_trend
from constants import DEFAULT_EXCLUDED_TOPICS

# דוגמה לשימוש
fig = plot_topics_trend(
    data, 
    topic_mappings, 
    min_year=1990,
    excluded_topics=[11, 15],  # להחריג נושאים 11 ו-15
    included_topics=None       # או לכלול רק נושאים ספציפיים
)
```

### 4. מבנה הקבצים / File Structure

```
lda_analysis/
├── constants.py              # קובץ קבועים חדש
├── lda_visualizations.py     # מעודכן עם סינון
├── generate_analysis_files.py # מעודכן עם ארגומנטים
├── streamlit_app.py          # מעודכן עם ממשק סינון
├── __init__.py
├── run_streamlit.sh
└── README_TOPIC_FILTERING.md # הקובץ הזה
```

### 5. דוגמאות שימוש מתקדמות / Advanced Usage Examples

#### דוגמה 1: ניתוח בלי נושאים "רעשיים"
```bash
# החרגה של נושאים שנחשבים לרעש או לא רלוונטיים
python generate_analysis_files.py --exclude-topics 11 15 20 25
```

#### דוגמה 2: ניתוח נושאים ספציפיים בלבד
```bash
# ניתוח רק של נושאים הקשורים לתחום משפטי מסוים
python generate_analysis_files.py --include-topics 1 3 7 12 18
```

#### דוגמה 3: ניתוח מתקדם עם פרמטרים מרובים
```bash
# ניתוח multi-topic עם סף גבוה, ללא נושאים ספציפיים, משנת 2010
python generate_analysis_files.py \
  --mode multi \
  --threshold 0.5 \
  --exclude-topics 11 15 20 \
  --min-year 2010

# ניתוח עם טופ-7 נושאים, סף נמוך, משנת 2000
python generate_analysis_files.py \
  --mode multi \
  --threshold 0.05 \
  --top-k 7 \
  --min-year 2000 \
  --exclude-topics 11
```

### 6. הערות חשובות / Important Notes

1. **ברירת מחדל**: אם לא מציינים ארגומנטים, נושא 11 יוחרג אוטומטית
2. **include vs exclude**: אי אפשר להשתמש בשניהם יחד - רק באחד מהם
3. **תוצאות ריקות**: אם הסינון מחריג יותר מדי נושאים, ייתכן שלא יהיו תוצאות
4. **ביצועים**: סינון נושאים מזרז את הניתוח ומפחית נפח הנתונים

### 7. פתרון בעיות / Troubleshooting

#### שגיאה: "No data remaining after topic filtering"
```bash
# בדוק שלא החרגת יותר מדי נושאים
python generate_analysis_files.py --exclude-topics 11  # במקום רשימה ארוכה

# או השתמש במצב הכללה
python generate_analysis_files.py --include-topics 1 2 3 4 5
```

#### שגיאה: Import errors
```bash
# ודא שאתה מריץ מהתיקייה הנכונה
cd lda_analysis
python generate_analysis_files.py
```

### 8. משוב ועדכונים עתידיים / Feedback and Future Updates

התיקונים כוללים:
- ✅ קובץ קבועים משותף
- ✅ סינון נושאים עם ארגומנטים
- ✅ ברירת מחדל להחרגת נושא 11
- ✅ תמיכה בטופ-K נושאים עבור multi-topic
- ✅ שינוי סף ברירת מחדל ל-0.1 במקום 0.3
- ✅ לוגיקה משולבת: סף + טופ-K (הטוב מבין השניים)
- ✅ עדכון כל המודולים
- ✅ ממשק גרפי ב-Streamlit

להצעות נוספות או דיווח על בעיות, פנה למפתח הקוד. 