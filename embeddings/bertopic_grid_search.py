import os
from itertools import product
from bertopic import BERTopic
from bertopic.backend._hftransformers import HFTransformerBackend
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline

from preprocessing.extract_content import get_documents_content
from embeddings.custom_embedders.mean_pooling_embedder import MeanPoolingEmbedder
from embeddings.custom_embedders.truncating_embedder import TruncatingEmbedder


def get_params():
    embedding_mapping = {
        'avichr/Legal-heBERT_ft': [MeanPoolingEmbedder, TruncatingEmbedder],
        'avichr/Legal-heBERT': [MeanPoolingEmbedder, TruncatingEmbedder],
        'onlplab/alephbert-base': [HFTransformerBackend, MeanPoolingEmbedder, TruncatingEmbedder],
        'dicta-il/alephbertgimmel-small': [MeanPoolingEmbedder, TruncatingEmbedder],
        'dicta-il/alephbertgimmel-base': [MeanPoolingEmbedder, TruncatingEmbedder],
    }
    use_ctfidf_transformer = [True, False]
    use_vectorizer = [True, False]
    min_topic_sizes = [5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300]
    parameter_combinations = product(embedding_mapping.keys(), use_ctfidf_transformer, use_vectorizer, min_topic_sizes)
    return parameter_combinations, embedding_mapping


def run_bertopic(data, directory_path,
                 custom_embedder, ctfidf_model,
                 vectorizer_model, min_topic_size):
    topic_model = BERTopic(embedding_model=custom_embedder, ctfidf_model=ctfidf_model,
                           vectorizer_model=vectorizer_model, min_topic_size=min_topic_size,
                           calculate_probabilities=True)
    topics, probs = topic_model.fit_transform(data)

    topic_model.save(directory_path, serialization="safetensors", save_ctfidf=True)


def run_bertopic_grid_search(data, base_directory, parameter_combinations, embedding_mapping, stop_words):
    for idx, params in enumerate(parameter_combinations):
        model_name, use_ctfidf, use_vector, min_topic_size = params

        custom_embedding_classes = embedding_mapping.get(model_name, [])

        for custom_embedding_class in custom_embedding_classes:
            directory_name = f"{model_name.replace('/', '_')}_custom{custom_embedding_class.__name__}_ctfidf{use_ctfidf}_vector{use_vector}_minsize{min_topic_size}/"
            directory_path = os.path.join(base_directory, directory_name)

            os.makedirs(directory_path, exist_ok=True)

            pipe = pipeline('feature-extraction', model=model_name, device='cuda:0')
            custom_embedder = custom_embedding_class(pipe)

            vectorizer_model = CountVectorizer(stop_words=stop_words) if use_vector else None

            ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True) if use_ctfidf else None

            run_bertopic(data, directory_path, custom_embedder, ctfidf_model, vectorizer_model, min_topic_size)

            print(f"Model {idx + 1} saved in directory: {directory_path}")


if __name__ == '__main__':
    directory_path = "/mnt/local/mikehash/Data/Nevo/NevoVerdicts"
    df = get_documents_content(directory_path)
    data = df['extracted_content'].values.tolist()

    base_directory = "/mnt/local/mikehash/Embeddings/Results"

    parameter_combinations, embedding_mapping = get_params()

    stop_words = open('../preprocessing/heb_stopwords.txt', 'r').read().split()

    run_bertopic_grid_search(data, base_directory, parameter_combinations, embedding_mapping, stop_words)
