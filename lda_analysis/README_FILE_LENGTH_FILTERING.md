# File Length Filtering for LDA Analysis
קיבת מסמכים לפי אורך עבור ניתוח LDA

## תוכן העניינים
- [סקירה כללית](#סקירה-כללית)
- [התקנה ותצורה](#התקנה-ותצורה)
- [שימוש](#שימוש)
- [קבצים ופונקציות](#קבצים-ופונקציות)
- [דוגמאות שימוש](#דוגמאות-שימוש)
- [פתרון בעיות](#פתרון-בעיות)

## סקירה כללית

מערכת סינון אורך הקבצים מאפשרת לסנן מסמכים קצרים מדי לפני ביצוע ניתוח LDA. זה עוזר לשפר את איכות הניתוח על ידי הוצאת מסמכים שאינם מכילים מספיק תוכן משמעותי.

### תכונות עיקריות:
- 📊 ניתוח קבצי דוגמה לקביעת סף מינימלי
- 📏 סינון אוטומטי של קבצים קצרים
- 🔧 תמיכה בממשק שורת פקודה ו-Streamlit
- 📈 דיווח מפורט על סטטיסטיקות הסינון
- ⚙️ אפשרות להשבתת הסינון לפי בקשה

## התקנה ותצורה

### שלב 1: ניתוח קבצי דוגמה
ראשית, צריך לבצע ניתוח של קבצי דוגמה כדי לקבוע את הסף המינימלי:

```bash
cd lda_analysis
python analyze_file_lengths.py
```

הסקריפט יחפש את הקבצים הבאים:
- `00003780-A03.txt`
- `00003890-I02.txt`
- `00003900-G02.txt`
- `00003920-A02.txt`
- `00003980-G03.txt`
- `00003990-V04.txt`

### שלב 2: עדכון קובץ הקבועים
הסקריפט יעדכן אוטומטי את `constants.py` עם הערכים הבאים:
- `FILE_LENGTH_MEDIAN`: מספר המילים החציון
- `FILE_LENGTH_MAXIMUM`: מספר המילים המקסימלי  
- `FILE_LENGTH_MIN_THRESHOLD`: סף מינימלי (מקסימום + 10)

### דוגמה לפלט הסקריפט:
```
📝 File Length Analysis for Supreme Court Dataset
============================================================
📁 Analyzing files in: /path/to/data/raw
🔍 Looking for 6 specific files...
✅ 00003780-A03.txt: 1,234 words
✅ 00003890-I02.txt: 987 words
✅ 00003900-G02.txt: 1,456 words
✅ 00003920-A02.txt: 876 words
✅ 00003980-G03.txt: 1,678 words
✅ 00003990-V04.txt: 1,123 words

📊 Statistics from 6 valid files:
📈 Median word count: 1,178
🔝 Maximum word count: 1,678
🎯 Threshold (max + 10): 1,688

✅ Updated constants file: /path/to/constants.py
🎉 Analysis completed!
📊 Files with fewer than 1,688 words will be filtered out in LDA analysis.
```

## שימוש

### שימוש בשורת פקודה

#### עם סינון אורך (ברירת מחדל):
```bash
python generate_analysis_files.py --mode both
```

#### ללא סינון אורך:
```bash
python generate_analysis_files.py --mode both --no-length-filter
```

#### עם פרמטרים נוספים:
```bash
python generate_analysis_files.py \
    --mode multi \
    --exclude-topics 11 15 \
    --threshold 0.15 \
    --top-k 5 \
    --min-year 1990
```

### שימוש ב-Streamlit

הפעל את אפליקציית Streamlit:
```bash
streamlit run streamlit_app.py
```

בממשק המשתמש:
1. בצד שמאל, תחת "📏 File Length Filtering"
2. סמן/בטל סימון של "Apply file length filtering"
3. הסטטוס יוצג:
   - ✅ אם הקבועים מוגדרים נכון
   - ⚠️ אם הקבועים לא מוגדרים

### שימוש פרוגרמטי

```python
from constants import check_file_length_constants
from file_length_filter import filter_lda_input_data, print_filtering_summary

# בדיקת הגדרות קבועים
constants_ok = check_file_length_constants()

# טעינת נתונים
doc_mappings = pd.read_csv("docs_topics.csv")
years_df = pd.read_csv("extracted_years.csv")

# יישום סינון
filtered_docs, filtered_years, stats = filter_lda_input_data(
    doc_mappings, years_df
)

# הצגת סיכום
print_filtering_summary(stats)
```

## קבצים ופונקציות

### קבצים עיקריים:

1. **`analyze_file_lengths.py`**
   - ניתוח קבצי דוגמה
   - חישוב סטטיסטיקות
   - עדכון קובץ הקבועים

2. **`file_length_filter.py`**
   - פונקציות שיתוף לסינון
   - אינטגרציה עם pipeline של LDA

3. **`constants.py`** (מעודכן)
   - קבועים לסינון אורך
   - פונקציות עזר

### פונקציות מרכזיות:

```python
# ניתוח קבצים
analyze_specific_files(file_names, data_dir)
calculate_statistics(word_counts)
update_constants_file(median, maximum, threshold)

# סינון נתונים
filter_lda_input_data(doc_mappings, years_df, min_word_count)
apply_length_filtering(df, min_word_count, add_word_counts)
add_word_counts_to_dataframe(df, text_files_dir)

# עזרים
get_file_word_count(file_path)
check_file_length_constants()
print_filtering_summary(stats)
```

## דוגמאות שימוש

### דוגמה 1: ניתוח בסיסי
```python
# 1. בצע ניתוח קבצי דוגמה
python analyze_file_lengths.py

# 2. הפעל ניתוח LDA עם סינון
python generate_analysis_files.py --mode both

# 3. הצג תוצאות ב-Streamlit
streamlit run streamlit_app.py
```

### דוגמה 2: מותאם אישית
```python
from file_length_filter import apply_length_filtering

# טען נתונים
df = pd.read_csv("my_data.csv")
df['word_count'] = df['filename'].apply(lambda x: get_file_word_count(Path(x)))

# יישם סינון מותאם
filtered_df, stats = apply_length_filtering(
    df, 
    min_word_count=500,  # סף מותאם
    add_word_counts=False
)

print_filtering_summary(stats)
```

### דוגמה 3: ללא סינון
```python
# השבת סינון אורך לביצוע ניתוח על כל הקבצים
python generate_analysis_files.py --no-length-filter --mode single
```

## פתרון בעיות

### בעיה: "FILE_LENGTH_MIN_THRESHOLD not set"
**פתרון:**
```bash
# הפעל את ניתוח אורך הקבצים
python analyze_file_lengths.py
```

### בעיה: "Data directory does not exist"
**פתרון:**
1. בדוק שקבצי הטקסט נמצאים במיקום הנכון
2. עדכן את הנתיב ב-`analyze_file_lengths.py`
3. אפשרויות נתיבים נפוצות:
   - `data/raw/`
   - `data/processed/`
   - `text_files/`

### בעיה: "No valid files found"
**פתרון:**
1. ודא שהקבצים הנדרשים קיימים:
   - `00003780-A03.txt`
   - `00003890-I02.txt` 
   - `00003900-G02.txt`
   - `00003920-A02.txt`
   - `00003980-G03.txt`
   - `00003990-V04.txt`
2. בדוק הרשאות קריאה לקבצים
3. ודא שהקבצים אינם ריקים

### בעיה: "File length filtering failed"
**פתרון:**
1. בדוק שעמודת `filename` קיימת בנתונים
2. ודא שנתיב הקבצים נכון
3. הפעל עם `--no-length-filter` כפתרון זמני

### בעיה: זיכרון גבוה בעת ספירת מילים
**פתרון:**
1. הסקריפט עובד על 6 קבצים דוגמה בלבד
2. אם הבעיה נמשכת, השתמש ב-`--no-length-filter`

## מידע טכני נוסף

### אלגוריתם ספירת מילים:
- פיצול לפי רווחים
- סינון מילים קצרות מ-2 תווים
- תמיכה בטקסט עברי ואנגלי
- ניסיון מספר קידודים שונים

### סף ברירת מחדל:
- מקסימום + 10 מילים
- מבטיח הכללת קבצים משמעותיים
- מסנן קבצים קצרים מדי

### ביצועים:
- ניתוח מהיר של 6 קבצי דוגמה
- cache של תוצאות ב-Streamlit
- עדכון אוטומטי של קובץ הקבועים

---

**לעזרה נוספת:** אנא פתח issue או צור קשר עם מפתח המערכת. 