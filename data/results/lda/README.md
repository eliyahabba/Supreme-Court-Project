# Supreme Court Topic Modeling Project - Data Structure

## 📋 Project Overview
Israeli Supreme Court verdict topic modeling project using LDA (Latent Dirichlet Allocation) analysis with preprocessing, topic modeling, and embeddings modules.

## 📁 Directory Structure

```
data/
├── raw_texts/                    # Input: Original verdict text files
│   └── [verdict_files].txt      # UTF-8 Hebrew text files
├── processed/                    # Processing results
│   └── extracted_years.csv      # Year extraction results
├── results/                      # LDA Analysis Results
│   ├── single_topic/            # Single-topic analysis (strongest topic per document)
│   │   ├── comprehensive_topic_data.csv
│   │   ├── processed_topic_data.csv
│   │   ├── yearly_topic_aggregation.csv
│   │   ├── yearly_statistics.csv
│   │   ├── topic_mappings_reference.csv
│   │   ├── topics_trend_single_topic.html
│   │   ├── topics_histogram_single_topic.html
│   │   └── topics_wordcloud_single_topic.png
│   └── multi_topic/             # Multi-topic analysis (all significant topics per document)
│       ├── multi_topic_data.csv
│       ├── yearly_topic_aggregation.csv
│       ├── yearly_statistics.csv
│       ├── topics_trend_multi_topic.html
│       ├── topics_histogram_multi_topic.html
│       └── topics_wordcloud_multi_topic.png
└── README.md                    # This file
```

## 🚀 Getting Started

### 1. **Prepare Input Data**
Place your Israeli Supreme Court verdict text files in `raw_texts/`:
- Text files in UTF-8 encoding
- Each file contains one verdict
- `.txt` extension required

### 2. **Extract Years from Documents**
```bash
# Basic usage - process all files
python preprocessing/extract_years.py

# Test with first 100 files
python preprocessing/extract_years.py --max-files 100

# Custom directories
python preprocessing/extract_years.py --input-dir /path/to/texts --output-dir /path/to/results

# Faster processing with more threads
python preprocessing/extract_years.py --max-workers 8
```

### 3. **Run LDA Analysis**
```bash
cd notebooks

# Run both analysis modes (default)
python lda_analysis.py

# Single-topic only (strongest topic per document)
python lda_analysis.py --mode single

# Multi-topic only (all significant topics per document)
python lda_analysis.py --mode multi
```

Or via Jupyter Notebook:
```bash
jupyter notebook "LDA Model Topic Analysis.ipynb"
```

## 📊 LDA Analysis Modes

### 🎯 **Single-Topic Analysis** (`results/lda/single_topic/`)
**Approach:** Each document gets assigned to its strongest topic only
- Simple and clear results
- Easy to interpret and visualize
- Traditional topic modeling approach
- Sum of percentages = 100%

### 🎯 **Multi-Topic Analysis** (`results/lda/multi_topic/`)
**Approach:** Documents can belong to multiple topics (threshold: 30% probability)
- More realistic representation of complex legal documents
- Reveals hidden connections between topics
- Simple counting: each document contributes to all significant topics
- Sum of topic counts can exceed number of documents

**Example:** A security-related case might score:
- 45% Security & Territories
- 23% Criminal Law  
- 15% Administrative Law
- Single-topic: Only "Security & Territories"
- Multi-topic: Only "Security & Territories" (others below 30% threshold)

## 📁 Output Files Explained

### 🎯 **For External Researchers - Key Files**

#### `comprehensive_topic_data.csv` (Single-Topic)
**Complete dataset with Hebrew topic descriptions**
- `filename` - Original file name
- `year` - Verdict year
- `strongest_topic` - Dominant topic number
- `strongest_topic_prob` - Confidence level (0-1)
- `topic_title` - Hebrew topic title (e.g., "חקירות משטרה", "ביטחון ושטחים")
- `topic_description` - Important words with weights

#### `multi_topic_data.csv` (Multi-Topic)
**Multi-topic dataset - all significant topics per document**
- `filename` - Original file name
- `year` - Verdict year
- `topic_id` - Topic number
- `topic_probability` - Topic probability in document (0-1)
- `is_strongest` - Whether this is the strongest topic (True/False)
- `topic_title` - Hebrew topic title
- `topic_description` - Important words with weights
- **Multiple rows per document** - one row per significant topic (>30%)

#### `topic_mappings_reference.csv`
**Topic reference guide**
- `topic_id` - Topic number
- `topic_title` - Hebrew topic title
- `topic_words` - Most important words for this topic

### 📊 **Analysis Files**

#### `yearly_topic_aggregation.csv`
**Annual aggregation by topics**
- `year` - Year
- `strongest_topic` - Topic number
- `verdicts` - Number of documents containing this topic (single) or document count (multi)
- `total_verdicts` - Total document appearances in that year
- `topic_percentage` - Topic percentage of total document appearances

#### `yearly_statistics.csv`
**Annual statistics**
- `year` - Year
- `total_verdicts` - Total document appearances
- `avg_topic_confidence` - Average confidence level
- `std_topic_confidence` - Standard deviation of confidence

### 🎨 **Visualizations**

#### `topics_trend_[mode].html`
**Interactive topic trends over time**
- Line chart showing how each topic changed over the years
- Choose/deselect specific topics
- Shows percentages of total document appearances per year
- Starts from 1948 (Israel's founding)
- Hebrew topic names in legend

#### `topics_histogram_[mode].html`
**Topic distribution**
- Bar chart of document count per topic
- Sorted from most to least common
- Shows dominant topics in the corpus
- Hebrew topic names on x-axis

#### `topics_wordcloud_[mode].png`
**Word clouds for all topics**
- Grid of word clouds (one per topic)
- Word size represents importance
- Hebrew font support
- Hebrew topic titles

## 🔍 Analysis Comparison

| Aspect | Single-Topic | Multi-Topic |
|--------|-------------|-------------|
| **Documents** | One topic per document | Multiple topics per document (>30%) |
| **Interpretation** | Simple and clear | Complex but realistic |
| **Counting** | One count per document | One count per significant topic |
| **Use Case** | General trends | Deep topic relationships |
| **Aggregation** | Simple counting | Simple counting (multiple per doc) |

## 🔬 Research Applications

### **Simple Analysis**
Use `comprehensive_topic_data.csv`:
- One document = one topic
- Clear temporal trends
- Easy statistical analysis

### **Advanced Analysis**
Use `multi_topic_data.csv`:
- One document = multiple topics (>30%)
- Complex legal document analysis
- Topic co-occurrence patterns

### **Combined Research**
Compare both modes:
- Run both analyses
- Compare trends between approaches
- Validate findings across methods

## 📋 Technical Requirements

### Dependencies
```bash
pip install gensim pandas plotly matplotlib wordcloud hebrew-wordcloud tqdm pathlib argparse
```

### Required Files
- LDA model files in `LDA Best Result/1693294471/`
- Custom topics file: `LDA Best Result/1693294471/topics_with_claude.csv`
- Extracted years: `data/processed/extracted_years.csv`

### System Requirements
- Python 3.8+
- Hebrew font support (macOS: SFHebrew.ttf)
- Sufficient memory for LDA model loading

## 🎯 Quick Start for Researchers

1. **Download the repository**
2. **Place verdict files** in `data/raw_texts/`
3. **Extract years:** `python preprocessing/extract_years.py`
4. **Run analysis:** `python notebooks/lda_analysis.py`
5. **Check results** in `data/results/single_topic/` and `data/results/multi_topic/`

## 💡 Tips

- **Interactive files (HTML)** open in browser with full interactivity
- **Start with single-topic** for initial understanding
- **Use multi-topic** for advanced research questions (30% threshold)