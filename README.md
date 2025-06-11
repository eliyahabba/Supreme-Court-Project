# Topic Modeling Of Verdicts

### Noa Bitan, Micha Hashkes

### Advisor: Dr. Renana Keydar

### October 1, 2023

## 1 Abstract

The vast amount of today’s textual data landscape, particularly in this study’s focus on legal
verdicts, poses a challenge in the absence of a tool capable of summarizing and analyzing
them. This study aims to tackle this issue and explore questions about Israeli law by using
Natural Language Processing tools. One of the key goals is to find methods that can extract
the main themes from a given verdict. These tools will output ranked clusters of key words
that describe topics, allowing law researchers to conduct further investigation. Furthermore,
this will enable the assessment of topic trends and their evolution, providing insight into the
changes within Israel’s Supreme Court over time.

## 2 Introduction

### 2.1 Background

In recent decades, Israel’s Supreme Court has issued tens of thousands of verdicts across a
range of categories. The aim of this research project is to utilize Natural Language Processing
(NLP) and Topic Modeling techniques to identify key topics within these documents. Upon
validation of these themes and their relevance, further inquiry can be conducted, such as
examining changes in thematic patterns over time.
Integrating NLP with the field of law can assist addressing various research inquiries
that may be beyond the scope of those unfamiliar with these tools. The corpus of available
documents is vast, and sifting through it can be challenging, particularly given that some
data may not be easily accessible to the average analyst.
This project seeks to use NLP techniques to reduce this complex and inaccessible legal
data into clusters of words that describe different topics, thus making it easier to query the
data by searching for specific keywords or categories. As previously mentioned, one of the
key areas of inquiry will be the evolution of word usage or patterns in topics over time.
Tools exist to address the challenge of discovering and annotating thematic information,
such as Latent Dirichlet Allocation (LDA), dynamic LDA, and more recently, transformer-
based models such as BERT. These models employ dimension reduction techniques on text
documents, identify connections throughout the data, and output patterns of words that


describe different topics. A key objective of this project is to investigate which model is most
suitable for analyzing the Supreme Court’s verdicts.

**2.1.1 Challenges of NLP in Hebrew**

Hebrew is a morphologically-rich language, meaning each input token may consist of multiple
lexical and functional units, each with a specific function in shaping the overall syntactic
or semantic structure. This implies that when dealing with Hebrew texts, the initial step
involves breaking down the Hebrew tokens into their constituent morphemes. Hebrew raw
tokens exhibit a high level of ambiguity. This complexity is increased by the absence of
diacritics ("accents") in standardized texts, resulting in the omission of most vowels. As
such, without context, no specific reading is inherently more probable than others. It is only
within context that the correct interpretation and segmentation become evident[1]
Furthermore, the absence of an abundant corpus of Hebrew text resources is a major
challenge to train and evaluate models. The Hebrew Wikipedia used most NLP research and
pre-training is orders of magnitude smaller than the English Wikipedia. Other resources are
mostly unlabeled and messy.

**2.1.2 Hebrew Transformers**

Extensive research efforts have been dedicated to addressing these challenges. One notable
development is AlephBert, a BERT-based model specifically tailored for morphologically-rich
languages. encompassing functions like Segmentation, Part-of-Speech Tagging, full morpho-
logical tagging, Named Entity Recognition, and Sentiment Analysis[2]. Building upon this,
the AlephBertGimmel model represents a further advancement[3]. Throughout this study,
transformer-based models were put to the test using pertinent data. Notably, the Legal-
HeBERT transformer stands out as a highly valuable pre-trained and fine-tuned model,
customized specifically for legal tasks[11].

**2.1.3 Lemmatization**

When it comes to probabilistic models, (in this case, LDA), the primary strategy for address-
ing the challenges posed by a morphologically-rich language lies in prioritizing the prepro-
cessing process. This involves tasks like tokenization and stop word removal. However, the
crucial step is lemmatization. Lemmatization is the process of reducing words to their base
or dictionary form, referred to as a lemma. This procedure aids in standardizing word forms
and streamlining text analysis by considering words in their most basic, canonical form. It
allows for a more focused approach on the core word itself, ensuring effective functioning in
algorithms that involve word counting.
The most commonly used lemmatization method used in Hebrew is a specialized tool
known as YAP. YAP conducts morphological analysis of input sentences, generating Part-of-
Speech tags, morphological features, lemmas, and more[1].


### 2.2 Related Work

**2.2.1 Topic Modeling**

Topic modeling is a popular field in NLP and already many research studies have been made
on the subject. One of the more important articles is "Latent Dirichlet Allocation"[7]. This
study introduces a generative probabilistic model called Latent Dirichlet Allocation (LDA),
as a method for constructing topics found in text corpora. The article compares LDA to
other similar models and highlights its advantages and disadvantages. It also explains how
to estimate the model’s parameters using a Bayesian approach.
Another significant advancement in this area involves clustering BERT-based embeddings
and deriving word representations from these clusters to formulate topics. This innovative
approach, known as BERTopic, is introduced and explained in the article "BERTopic: Neural
topic modeling with a class-based TF-IDF procedure"[8].

**2.2.2 Topic Modeling in Legal Data**

Topic modeling is becoming increasingly relevant and utilized within the legal field for in-
depth analysis. For instance, the study "Modeling Repressive Policing" employs LDA to
map and analyze topics in the protocols of inquiries into lethal clashes between the Israeli
police and Israels Arab minority in October 2000[9]. Additionally, "Topic Modeling of Legal
Documents via LEGAL-BERT" attempts to employ embeddings for modeling topics in US
legal text[10].

### 2.3 Data

This study is based on an extensive dataset comprising thousands of verdicts from Israel’s
Supreme Court. A predefined set of criteria for this project included selecting verdicts that
exceed 2 pages in length and specific procedure types, including those of the HCG.
Although the Supreme Court website[4] contains verdicts within their domain, there is
no readily available API for extraction. The Nevo website[5] is another platform containing
various accessible legal data through queries. An initial scraping tool was developed to
extract the relevant query, resulting in over 30,000 verdicts converted into text documents.
Unrelated to this project, a database containing all of the Supreme Court’s legal docu-
ments, along with a wealth of metadata, was uploaded to the HuggingFace platform[6] during
the course of this research. While this dataset proved to be more user-friendly and faster
for navigation and querying, it presented some discrepancies in numbers when compared to
the previously extracted data. Additionally, it was missing data from before 1997. As it
was important to evaluate trends through decade of legal data, it was decided to utilize the
verdicts from Nevo, applying an additional filter for those exceeding 500 words, ultimately
yielding over 11,000 verdicts.
The data, for the most part, is unstructured. Although there is a discernible pattern to
how verdicts are written, not all of them adhere to the conventional structure. Elements such
as date, judges, and verdict content are not consistently located in obvious places. Therefore,
information like content and the years in which the verdicts were issued were extracted using
regular expressions.


```
Figure 1: Histogram of Years verdicts were given in
```
## 3 Results

In this research, two primary experiments were conducted. The first utilized the LDA model,
while the second employed the BERTopic algorithm. This section will elaborate on the various
results obtained from these two approaches. An in-depth explanation of both models and
how they function is provided later in this paper.

### 3.1 Latent Dirichlet Allocation

As a probabilistic model, Latent Dirichlet Allocation is less sensitive to sentence structure and
places greater emphasis on word distributions. Therefore, the data underwent tokenization
and lemmatization.

**3.1.1 LDA Pipeline**

As an initial solution to the research question, a comprehensive LDA pipeline was developed.
This pipeline has the capability to process text files from any source and allows for a specified
range of parameter values. It conducts a grid search within this parameter space, comput-
ing coherence scores for each model. The pipeline is designed to be user-friendly, enabling
easy and efficient execution, streamlining the process of LDA fine-tuning, and ensuring its
convenience and robustness.
Using the grid-search technique, theαandβparameters were varied between 0.1 and
4, while the number of topics ranged from 20 to 35. These values were carefully selected
after exploring different ranges and observing the behavior of coherence scores. With this
configuration, over 4000 models were trained, each with a unique combination of parameters.


**3.1.2 LDA Word Analysis**

In the LDA output, each "topic" is represented as a list of words, each accompanied by its
weight within the topic. As part of the LDA analysis, it was important to determine the
optimal number of words that should define a topic. To examine this, the top 50 words for
each topic were selected, and a cumulative sum of probabilities was calculated for each index.
The findings revealed that using the top 25 words covers approximately 25% of the topic,
while with 400 words, around 80% coverage was achieved. Interestingly, a significant change
in trend behavior occurs at around 150 words, as it transitions to a more linear pattern.

```
Figure 2: Average Cumulative Sum Of Probabilities Per Index - Top 50
```
```
Figure 3: Average Cumulative Sum Of Probabilities Per Index - Zoom out
```

**3.1.3 Cv-Coherence Analysis**

The Cv-coherence score, which is a well used coherence score for evaluating an LDA model,
is interpreted such that higher values suggest a potentially better model.
The highest recorded Cv score was 0.54, achieved with 35 topics, anαvalue of 0.6, and
anβvalue of 0.7. After analysis of grid search results, it appears thatβparameter has a
relatively minor impact on the Cv score, compared to theαparameter. Notably, for lower
values ofα (ranging from 0.1 to 1), it seems that the Cv score tends to be higher. This
conclusion can be seen on the accompanying graphs.

```
Figure 4: Cv-Coherence score vs.α
```
```
Figure 5: Cv-Coherence score vs.β
```

In addition, when examining models with 30 or 35 topics, it becomes evident that theα
parameter exerts a more pronounced influence on the Cv score.

```
Figure 6: Cv Heatmap
```
In the highlighted points of the graph above, the upper 10th percentile is emphasized.
Lower values ofαtend to yield better models, while theβparameter appears to have a less
significant impact. Theαparameter signifies the distribution of topics within a document,
and a lowerαvalue indicates that a document concentrates on a smaller set of topics. This
aligns with the nature of the observed documents, which are Supreme Court verdicts, each
dealing with a specific case.
These findings pose a challenge in terms of model selection, as there is a considerable
amount of noisy points. From the figure, it seems worthwhile to focus onαranging from 0
to 1. However, it’s important to note that there are still some good models outside of this
range.

**3.1.4 UMass-Coherence Analysis**

The UMass coherence score is another widely used measurement method, often regarded
as superior to Cv. UMass scores are expressed as negative values due to the logarithmic
calculation involving values smaller than 1. The higher the UMass score, the better the
model. However, when examining various ranges ofα, β, and their relationship with the
UMass score, it seems inconsistent.


```
Figure 7: UMass Heatmap - Upper 10th percentile
```
In the graph above, the highlighted points represent the upper 10th percentile based on
their UMass score. As can be seen, there is no discernible concentration of these points
within any specific region of the parameter space.
Upon closer examination of the upper 1st percentile, a significant distinction emerges
between models with 25 topics and those with 35 topics. For models with 25 topics, the
most optimal configurations for alpha are found within the range of 0 to 1.1. In contrast, for
models with 35 topics, the best-performing models exhibit alpha values ranging from 2 to 4.
One possible explanation for this observation could be attributed to the nature of the topics
themselves. In a model with 25 topics, each topic may encompass a broader scope, leading
documents to focus on a smaller set of topics. In contrast, in a model with 35 topics, the
topics may be more specific, allowing documents to potentially encompass a larger array of
topics.


```
Figure 8: UMass Heatmap - 25 topics
```
```
Figure 9: UMass Heatmap - 35 topics
```
**3.1.5 LDA Model Analysis**

The best LDA model in terms of coherence achieved a UMass-coherence score of -0.88, with
anαvalue of 0.2,βset at 2.9, and 25 topics selected. Some of the topics are very broad and
too general, while others are more specialized, delving into specific subjects such as divorces,
human rights, prisoners, and more.


```
Figure 10: Word Cloud Per Topic, LDA Model
```
It is also interesting to examine the topic trends. For example, there is a notable surge
in the frequency of the "West Bank" topic during the 2000s. This pattern can be attributed
to the high incidence of terrorist activities during those years, as well as the negotiations
surrounding the West Bank barrier.

```
Figure 11: West Bank Topic Trend
```

### 3.2 Embeddings - BERTopic

As previously mentioned, token embeddings can also serve as a foundation for Topic Model-
ing. This involves clustering the embeddings, identifying representative documents (referred
to as "exemplars") for each cluster, and ultimately deriving topics by associating documents
with their closest exemplar. An algorithm that employs this approach is BERTopic, which
leverages any BERT-based transformer to scrutinize the embeddings.

**3.2.1 BERTopic Pipeline**

A pipeline for BERTopic, similar to LDA, was developed to facilitate the analysis of legal ver-
dict topics. This algorithm does not heavily emphasize hyper-parameters; instead, it allows
the user to choose the model for each step of the algorithm (e.g., embedding model, dimen-
sionality reduction method, clustering method, TF-IDF model, etc.). Unlike LDA, BERTopic
does not require preprocessing and it is even discouraged, as contextual information is cru-
cial for transformer embeddings. However, after experimenting with various variations, it
appears that there is a necessity to remove stop words during the topic extraction stage.
Another challenge arises from the fact that most BERT-based models nowadays have a
maximum token limit of 512. This poses a problem for many legal verdict documents, which
often exceed this limit. To address this issue, two custom embedding methods were developed
in this project in addition to the original BERTopic algorithm. The first method involves an
embedder that takes the document and truncates it to the first 512 tokens. Upon manual
inspection of some verdicts, it was observed that the document’s primary topic typically
emerges within the initial paragraph. However, this truncation may lead to the omission
of crucial context, potentially resulting in the loss of key terms. The second method is
mean pooling, which entails taking chunks of 512 tokens and computing their average. This
approach captures the complete context but may result in a loss of finer details.
As a result, there exist numerous ways to implement this algorithm. Like with LDA, a grid
search was developed, considering parameters such as which embedding models to employ
(e.g., AlephBert, LegalHeBERT, AlephBertGimmel), which custom embedding method to
use, whether to reduce common words, whether to eliminate stop words, and the minimum
topic size (i.e., the number of documents containing a topic). Regarding the selection of the
number of topics, it is not recommended to do so, as one of the algorithm’s functions is to
merge topics that are similar.

**3.2.2 BERTopic Evaluation**

Assessing the BERTopic algorithm is not as straightforward as evaluating probabilistic mod-
els. While coherence scores can be utilized, they are less effective in this context. This
is because, instead of a bag-of-words representation, the embeddings are more contextually
aware, making coherence scores less applicable. Furthermore, the primary objective here
is focused on unsupervised classification and generating diverse topics, rather than strictly
producing coherent words. Due to a constraint of time and lack of a better way, the chosen
approach involved identifying models that yielded a similar number of topics as the best LDA
models and manually evaluating the clarity of the topics. In later sections of this paper, more
refined evaluation methods will be discussed.


**3.2.3 BERTopic Topic Analysis**

By employing a model with over 30 topics, using the LegalHeBERT model[11], along with the
utilization of the truncating custom embedder and the removal of stop words, BERTopic suc-
cessfully forms well-defined clusters for a range of both common and specialized topics. While
some clusters encompass broad and general themes characterized by legal document termi-
nology, the majority offer insight into the specific subjects addressed within these verdicts.
Figure 12 displays the most frequent topics along with the associated probability-weighted
words that define them. Figure 13 shows the visualization of the topics in a two-dimensional
space, while the area of these topic clusters is proportional to the amount of words that
belong to each topic across the dictionary.

```
Figure 12: BERTopic Topic Representations with c-TF-IDF Scores
```

```
Figure 13: BERTopic Intertopic Distance Map
```
**3.2.4 Dynamic Topic Modeling**

BERTopic’s algorithm enables dynamic topic modeling, which involves analyzing how topics
evolve over time and understanding their representation across different time periods. This is
achieved by initially fitting BERTopic without considering the temporal aspect of the data.
Then, the representative documents for each topic are segmented into different time steps,
corresponding to the years in which the verdicts were issued. Then, for each topic and time
step, the c-TF-IDF representation is computed, yielding a specific topic representation for
each time period.
This approach often yields more insightful results than analyzing topic trends, as can be
done with LDA. It provides a nuanced perspective on how a topic is utilized differently over
time. This table illustrates the evaluation of topics such as government and security across
various time periods.

```
Topic Year Words
Government/Elections 1985 !רקובמMינגסהותסנכההריקחרקבמה
Government/Elections 2005 !יקסבולורוגMייונימהתסנכתסנכהתוניסחה
Government/Elections 2021 !תננוכמההיציזופואהביצקתתסנכהביצקתה
Government/Elections 2022 !יקסבוקיליממהאוולההתונתמהZרפוהינתנ
Security/Terror 2004 !MיללחהMינומלאהתוחפשמהרבקה
Security/Terror 2009 !MתיילעMירקבמתבשותיבהרהל
Security/Terror 2022 !עגפמהדקפמהעוגיפהלבחמה
```

```
Figure 14: BERTopic Dynamic Topic Modeling Frequency (Government & Security)
```
**3.2.5 Hierarchical Topic Modeling**

Another intriguing application of word clustering is hierarchical topic modeling. This ap-
proach aims to capture the potential hierarchical structure among the created topics, pro-
viding insights into which topics share similarities and offering a deeper understanding of
potential sub-topics within the dataset. This hierarchy can be approximated using the c-TF-
IDF matrix, which contains information about the significance of each word in every topic.
The smaller the distance between two c-TF-IDF representations, the more similar the topics
are.
When combining two topics, their c-TF-IDF representation can be recalculated by ag-
gregating their bag-of-words representation. This allows for the examination of topic repre-
sentations at each level in the hierarchy. As depicted in Figure 15, not all hierarchies are
interesting or even logical, but some can give insight to how the topics are similar.

```
Figure 15: BERTopic Hierarchical Topics Clustering
```

**3.2.6 LDA vs. BERTopic**

After selecting a model for each of the method, 4 pairs of similar topics were selected, as
shown in the figure below.

```
Figure 16: Word Cloud Comparison. First row is BERTopic, second LDA
```
In general, it appears that both methods yield similar key terms for each topic. In case of
BERTopic, unprocessed text data can lead to topics containing numerical values as prominent
terms. Further distinguishable differences will be discussed in a later section.

## 4 Discussion

### 4.1 Summary of Results in Retrospective

As previously noted, topic modeling is not a new field Natural Language Processing. Estab-
lished models such as LDA have already been tried and tested, proven to mostly work as
intended. In this research, similar positive outcomes were observed, as discernible word pat-
terns were successfully mapped across the dataset, although domain experts should conduct
further verification. This alignment with established findings is not unexpected, as many
studies, including those in legal data research and across various languages, have employed
probabilistic models for similar objectives successfully.
In contrast, the primary innovation in this research lies in the application of a less com-
monly utilized method, the use of word embeddings, which has also yielded satisfactory
results. Over the past few years, significant efforts have been directed towards enhancing the
capabilities of sentence encoders and embeddings. This development has paved the way for
tackling various NLP tasks in a new and efficient manner. This study serves as evidence that
this approach can also yield commendable results in the realm of topic modeling for legal
verdicts, thanks to the implementation of BERTopic’s algorithm. This achievement holds


considerable significance, as it demonstrates the potential for more confident utilization of
this method in the future, particularly in Hebrew, where the field of transformers has only
recently gained momentum.

**4.1.1 LDA Results in Retrospective**

In general, LDA models appear to have efficiently produced satisfactory results and are
straightforward to set up. The capability to manage hyper-parameters and predict their
impact enables the execution of massive experiments and the analysis of parameter trends.
It seems there is still a room for further investigation of hyper-parameters space, with the
aim of achieving better results. LDA results can serve as a decent baseline for other type of
models as well.
Based on the conducted experiment, it was found that the optimal number of topics is
approximately 30. However, the ultimate decision on the number of topics should be left to
a domain expert in the legal field, depending on the specific research question. The LDA
pipeline’s user-friendly configuration and flexibility enable thorough exploration to cater to
different domain-specific requirements.
Throughout the LDA training process, there was a trade-off between efficiency and accu-
racy. This trade-off reflected in choosing the values for "iterations" and "passes" parameters.
Passes controls how many times the entire corpus is processed during training. While the
Gensim default value is 1, it seems that in this case 5 is preferable. Iterations controls how
many times the model updates its topics for each document in a single pass, and the Gensim
default value for it was implemented. Perhaps using higher values for these two parameters
may produce better results, but with higher complexity.

**4.1.2 BERTopic Results in Retrospective**

BERTopic proves to be remarkably versatile and adaptable, enabling the use of distinct
sub-models and parameters at each step. When employing the default parameters from the
built-in Python BERTopic library, the results were suboptimal, primarily identifying only
the most common words as topics. Nonetheless, this flexibility allowed for the substitution
of the default embedding model with a Hebrew-trained BERT encoder. This adjustment led
to significantly improved and coherent topics that possess meaningful interpretations.
This algorithm offers even greater versatility by allowing for experimentation with the
word clusters, enabling the discovery of more substantial and nuanced results. Techniques
like dynamic topic modeling and hierarchical topic analysis become easier, providing deeper
insights into the data and enhancing the potential for further analysis. Dynamic modeling is
particularly intriguing, especially in the current political climate, as it can reveal how judges
are currently addressing topics in contrast to the past.

**4.1.3 Comparing LDA and BERTopic Models**

One of the key goals of this research was to assess and compare various topic modeling meth-
ods to determine which one performed more effectively on the Israel Supreme Court verdicts.
In hindsight, this proved to be a challenging task. Contemporary evaluation methods like
coherence scores offer limited insight for meaningful comparisons. Instead, a viable approach


was to compare the outputs of two models - one from LDA and the other from BERTopic.
This involved examining the shared topics, distinctions in coverage, and the specific termi-
nology employed by each model.
The most notable distinction between the two methods is that LDA necessitates pre-
processing the data into lemmas, whereas embeddings do not require any preprocessing.
Consequently, in embeddings, there is a tendency for repeated words in the same topic that
share the same base but differ in morphological structure within a sentence. On the other
hand, with LDA, the results permit a more varied use of words to describe a particular topic,
as opposed to relying on the same word with differing variations.
Another key difference between the two is that BERTopic and transformer-based models,
in general, use the attention mechanism - the ability to zoom in into a word in a sentence,
along with it’s relationship with the rest elements in sentence. LDA treats every word as an
individual token, without taking into account it’s context.
Regarding interpretability, LDA outcomes are more straightforward to comprehend com-
pared to BERTopic, which relies on neural networks for its operations.

### 4.2 Conclusions Based on the Results

The algorithms applied in this research have achieved results that can enhance and facilitate
the work of legal analysts in conducting their own research, highlighting pertinent topics
within today’s legal landscape. Moreover, it is reasonable to assert that with further experi-
mentation, the integration of transformer models and embeddings in NLP techniques holds
promising potential for effectively analyzing legal data in Hebrew.

### 4.3 Limitations

**4.3.1 NLP in Hebrew**

Starting off, it’s important to note that working with Hebrew in the field of Natural Language
Processing is a complex task, and it’s an area that continues to evolve. In the thought process
of this project, the limitations were considered, including the fact that numerous cutting-edge
tools that work well in other languages cannot be applied to Hebrew.

**4.3.2 Model Evaluation**

In this context, it’s important to note that the topic modeling task is unsupervised, meaning
the corpus lacks a known or true topics distribution. Although Gensim’s library offers a
built-in mechanism for efficiently calculating coherence scores, it’s worth mentioning that
these scores may not necessarily be the most comprehensive way to evaluate the models. A
reliable method of evaluation involves having legal experts review the topics to assess their
relevance. However, this approach is also challenged by resource constraints.


### 4.4 Future Work

**4.4.1 Model Evaluation**

As mentioned, there is a room for investigating different evaluating methods rather then
coherence scores, such as perplexity. In addition, there are some approaches which involves
human opinion, such as word intrusion and topic intrusion [14]. In this context, it can be
a good idea to involve domain-expert when evaluating model quality, in such a way that
ranking and feedback process will be easy and fast.
An insight that came along while working on the project was the creation of unique score
measure, according to domain-expert guidelines.

## 5 Resources and Methods

### 5.1 Preprocessing

**5.1.1 Nevo Scraping**

In order to retrieve the relevant data, a script was developed using the Selenium Python
package^1. This script was designed to systematically retrieve all the legal verdicts from a
predetermined query on the Nevo website. Given the website’s limitation of processing no
more than 10,000 documents at once, the query was partitioned in the code into smaller,
manageable segments. After the documents were retrieved, these PDF files were converted
into text documents.

**5.1.2 Content Extraction**

The actual content of the verdict is surrounded by text that is unnecessary for the topic
models. As a result, regular expressions were used to identify specific words indicating the
beginning and end of the content. The accuracy of this process was manually validated
through sampling and was found to be effective in about 90% of the documents.
Additionally, regular expressions were used to extract dates in order to determine the years
in which the verdicts were issued. This information proves valuable for dynamic modeling and
evaluating topics over time. Given that a document may reference various years, including
incidents from the past without a standardized placement of an official date within the text,
the highest year mentioned was selected.

**5.1.3 Lemmatizing**

Initially, the Trankit library, which supports Hebrew lemmas, was tested. However, the
quality of the lemmatized text did not meet expectations. In contrast, YAP (Yet Another
Parser)[7] proved to be a significantly superior lemmatizer due to its effective handling of
Hebrew morphology.

(^1) based loosely on another student’s (Noam Maeir) script


### 5.2 Statistical Methods and Considerations

In order to measure topic modeling model, a common and straight-forward method is to
use a coherence score. Typically, coherence scores reflect the degree of coherence among the
various topics generated by the model.
Coherence scores are computed individually for each topic, and the overall model perfor-
mance is determined by averaging these scores. There are some different coherence scores,
and the ones used in this project were CV and UMass scores.

**5.2.1 UMass score**

According to ’Exploring the Space of Topic Coherence Measures’[12], UMass score formula
is given by

```
2
N·(N−1)
```
#### ∑N

```
i=
```
```
∑i−^1
```
```
j=
```
```
log
```
```
P(wi, wj) +
P(wj)
```
, Where the term
P(wi, wj)
P(wj)

is equivalent to the probability of viewing the wordwi, along with the wordwj. Here, N
represents top words per topic, and as mentioned, in Gensim implementation the default is
N=20. Since the term inside the log is a fraction smaller then 1 (whenis small), UMass
score sums negative numbers, therefore it’s value is all value negative.

**5.2.2 CV score**

In order to explain CV score, first it’s important to understand PMI - pointwise mutual
information. PMI is a measure of co-occurance between two words, and is given by the
following formula:

```
log
```
P(wi, wj) +
P(wj)(wi)
In order to calculate probabilities, CV score uses sliding window calculation - for a fixed
window size (110 words), the termP(wi, wj)stands for the amount of sliding windows the
two appear, out of all possible sliding windows in corpus.
For each topic, there are N word-topic vectors. Each vector corresponds to a word, and
contains it’s NPMI (normalized PMI) value with the rest of the words in topic. Then, in order
to create a single vector for each topic, CV sums the N vectors for. Using cosine operation,
a similarity measure is created between topic vector and word-topic vector. The topic CV
score is calculated as the average of these cosine similarities, and the overall CV score for the
entire model is derived by averaging the scores across all topics.


### 5.3 Models and ML Methods Used

#### 5.3.1 LDA

Latent Dirichlet Allocation (LDA) is an unsupervised probabilistic generative model, used
for topic modeling. It is a statistical model that represents documents as mixtures of topics,
where each topic is characterized by a distribution over words. LDA assumes that docu-
ments are created through a two-step generative process involving topics and words. LDA
hyper-parameters are number of topics,αandη. Bothα andηare hyper-parameters for
the Dirichlet distribution, α relates to topic-document distribution, andη for word-topic
distribution.
The main idea behind the LDA algorithm is to maximize seen corpus likelihood, given
configured hyper-parameters. Model assumption are that topic-document and word-topic
distributions belongs to Dirichlet distributions, and order of words in a document doesn’t
matter to topic assignment. The original inferential problem of the LDA, given a specific
document is:

```
p(θ, z|w, α, β) =
```
p(θ, z, w|α, β)
p(w|α, β)
Whereθrepresents topic distribution inside the document andzrepresents topic assign-
ment to each word in document. In David Blei’s article[7], it’s explained why this formula
can’t be calculated, and LDA implementations use approximations for this.

```
Figure 17: LDA model, taken from David Blei’s article[7]
```
**5.3.2 BERTopic**

BERTopic is a topic modeling algorithm that relies on clusters formed by transformer-based
embeddings. The algorithm follows these steps:

1. Embeddings: Converting documents to numerical representations. This is by default
    sentence-transformers, but any embedding model is supported, as long as it fits the
    relevant case.


2. Dimensionality reduction: Reducing the dimensionality of the embedded representa-
    tions, to fit cluster models that have difficulty handling high dimensional data. The
    proposed technique used is UMAP, as it effectively preserves both local and global
    structure while reducing the dataset’s dimensionality, thus enabling the discovery of
    similarity between documents.
3. Clustering: Applying a clustering method on the reduced embeddings. The proposed
    model is HDBSCAN, a density-based clustering technique, as it can find clusters of
    various shape, identifies outliers and noise, and does not need pre-specifying the number
    of clusters.
4. Bag-of-words representation: Combining all documents in a cluster into a single docu-
    ment, and counting how often each word appears in each cluster. This L1-normalized
    bag-of-words representation operates at the cluster level, allowing topics to be generated
    from clusters rather than individual documents. A straightforward CountVectorizer can
    be employed for this purpose.
5. Topic representation: using c-TF-IDF to find topic representations unique to each
    cluster. This class-based TF-IDF procedure models the significance of words within
    clusters, rather than individual documents.

```
Figure 18: BERTopic Pipeline[8]
```
As previously mentioned, this algorithm underwent several experiments with various pa-
rameters and sub-models. One of the most challenging aspects was identifying the appropriate
embedding model to suit the specific data of this project. Initially, there was promise in fine-
tuning the AlephBert model, which is well-regarded in the Hebrew NLP community, with the
extracted legal verdicts. However, using an already fine-tuned model specifically designed
for legal data, known as LegalHeBERT, proved to be more advantageous. The other recom-
mended sub-models in the algorithm (UMAP, HDBSCAN, c-TF-IDF) were also employed,
with the exception of using a bag-of-words approach that excluded Hebrew stopwords.


### 5.4 Resources

**5.4.1 Project Code**

The project’s code is in Git: https://github.com/noabitan1/DS-Research-Project
The code contains three modules, each used separately:

1. preprocesseing
2. lda
3. embeddings

**5.4.2 External Resources**

- ONLP YAP - https://github.com/OnlpLab/yap
- YAP Python Wrapper - https://github.com/amit-shkolnik/YAP-Wrapper
- Gensim Python Library (for LDA, Coherence) - https://radimrehurek.com/gensim/
- Hebrew Stop words - https://github.com/gidim/HebrewStopWords
- BERTopic https://maartengr.github.io/BERTopic/index.html
- HuggingFace Transformers https://huggingface.co/docs/transformers/index
- Some reference notebooks and scripts sent in the beginning by Noam Maeir [maeirnoam@gmail.com]
- ChatGPT - for helping when getting stuck in code, and for imporving the grammar in
    this paper


### 5.5 Work Pipeline

```
Figure 19: Work Pipeline
```
This project primarily addressed two significant directions: LDA and embeddings; however,
the preprocessing was not the same. While the initial phases involved data scraping and
content extraction for both, embeddings did not require text to undergo lemmatization or
cleaning.
The capability of grid search in LDA was added after numerous experiments and the
understanding that to attain a high-quality model, it is essential to have a robust method
for running lots of models with various hyper-parameter variations.
Cleaning and filtering stage involved removing non-Hebrew words, numbers, and after
lemmatization - single letter words which don’t contribute to topic modeling task.
In the model evaluation stage, the gradual inclusion of coherence score calculations was
integrated into the pipeline.
A grid search for BERTopic was also developed, but the focus was more on finding the
right sub-models for the data manually, rather than letting an automatic process evaluate
hyper-parameters.

### Division of work

- Noa Bitan: Preprocessing, LDA


- Micha Hashkes: Preprocessing, BERTopic

## References

[1] Reut Tsarfaty, Amit Seker, Shoval Sadde, Stav Klein. Whats Wrong with Hebrew NLP?
And How to Make it Right. Open University of Israel. https://github.com/OnlpLab/yap

[2] Amit Seker, Elron Bandel, Dan Bareket, Idan Brusilovsky, Refael Shaked Greenfeld, Reut
Tsarfaty. AlephBERT: A Hebrew Large Pre-Trained Language Model to Start-off your
Hebrew NLP Application With. Bar-Ilan University

[3] Eylon Gueta, Avi Shmidman, Shaltiel Shmidman, Cheyn Shmuel Shmidman,Joshua
Guedalia, Moshe Koppel, Dan Bareket, Amit Seker1, Reut Tsarfaty. Large Pre-Trained
Models with Extra-Large Vocabularies: A Contrastive Analysis of Hebrew BERT Models
and a New One to Outperform Them All. Bar Ilan University, DICTA

[4] Israel Supreme Court, https://supreme.court.gov.il/Pages/HomePage.aspx

[5] Nevo, https://www.nevo.co.il/

[6] Lev Muchnik, Inbal Yahav, Ariel Nevo, Avichay Chriqui, Tim Shektov, 2023, The
Israeli Supreme Court Dataset. https://huggingface.co/datasets/LevMuchnik/Supreme-
CourtOfIsrael

[7] Blei, Ng, Jordan. Latent Dirichlet Allocation. Journal of Machine Learning Research,

2003. https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf

[8] Maarten Grootendorst. BERTopic: Neural topic modeling with a
class-based TF-IDF procedure. https://doi.org/10.48550/arXiv.2203.05794
https://maartengr.github.io/BERTopic/index.html

[9] Renana Keydar, Yael Litmanovitz, Badi Hasisi, Yoav Kan-Tor. Modeling Repressive
Policing: Computational Analysis of Protocols from the Israeli State Commission
of Inquiry into the October 2000 Events. Law & Social Inquiry, 47(4), 1075-1105.
doi:10.1017/lsi.2021.63

[10] Silveira, Fernandes, Neto, Furtado, Pimentel Filho, José Ernesto, Topic
Modelling of Legal Documents via LEGAL-BERT (June 25, 2021).
[http://dx.doi.org/10.2139/ssrn.4539091](http://dx.doi.org/10.2139/ssrn.4539091)

[11] Chriqui, Avihay and Yahav, Inbal and Bar-Siman-Tov, Ittai, Legal HeBERT: A
BERT-based NLP Model for Hebrew Legal, Judicial and Legislative Texts (June
27, 2022). Bar Ilan University. [http://dx.doi.org/10.2139/ssrn.4147127](http://dx.doi.org/10.2139/ssrn.4147127) https://hugging-
face.co/avichr/Legal-heBERT

[12] Michael, Röder, Andreas, Both, Alexander, Hinneburg, Ex-
ploring the Space of Topic Coherence Measures (2015)
https://svn.aksw.org/papers/2015/WSDMTopicEvaluation/public.pdf


[13] Emil Rijcken, Pablo Mosteiro, Kalliopi Zervanou, Marco Spruit, Floortje Scheepers, Uzay
Kaymak https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=arnumber=9882661

[14] Jonathan Chang, Jordan Boyd-Graber,Sean Gerrish, Chong Wang, David M. Blei. Reading
Tea Leaves: How Humans Interpret Topic Models https://www.cs.columbia.edu/ blei/pa-
pers/ChangBoyd-GraberWangGerrishBlei2009a.pdf


