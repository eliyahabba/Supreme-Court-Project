import numpy as np

from tqdm import tqdm
from typing import List
from sklearn.preprocessing import normalize
from transformers.pipelines import Pipeline

from bertopic.backend import BaseEmbedder


class CustomEmbedder(BaseEmbedder):
    def __init__(self, embedding_model: Pipeline):
        super().__init__()

        if isinstance(embedding_model, Pipeline):
            self.embedding_model = embedding_model
        else:
            raise ValueError("Please select a correct transformers pipeline. For example: "
                             "pipeline('feature-extraction', model='distilbert-base-cased', device=0)")

    def embed(self,
              documents: List[str],
              verbose: bool = False) -> np.ndarray:
        """ Embed a list of n documents/words into an n-dimensional
        matrix of embeddings

        Arguments:
            documents: A list of documents or words to be embedded
            verbose: Controls the verbosity of the process

        Returns:
            Document/words embeddings with shape (n, m) with `n` documents/words
            that each have an embeddings size of `m`
        """
        max_length = 512  # Set the maximum length to 512 tokens

        embeddings = []
        for document in tqdm(documents, total=len(documents), disable=not verbose):
            # Use the tokenizer provided by self.embedding_model
            tokens = self.embedding_model.tokenizer(document, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
            decoded_document = self.embedding_model.tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)
            truncated_document = document[:len(decoded_document) - 100]

            # Get embeddings using the embedding model
            features = self.embedding_model(truncated_document, truncation=True, padding=True)

            embeddings.append(self._embed(truncated_document, features))

        return np.array(embeddings)


    def _embed(self,
               document: str,
               features: np.ndarray) -> np.ndarray:
        """ Mean pooling

        Arguments:
            document: The document for which to extract the attention mask
            features: The embeddings for each token

        Adopted from:
        https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2#usage-huggingface-transformers
        """
        token_embeddings = np.array(features)
        attention_mask = self.embedding_model.tokenizer(document, truncation=True, padding=True, return_tensors="np")["attention_mask"]
        input_mask_expanded = np.broadcast_to(np.expand_dims(attention_mask, -1), token_embeddings.shape)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = np.clip(input_mask_expanded.sum(1), a_min=1e-9, a_max=input_mask_expanded.sum(1).max())
        embedding = normalize(sum_embeddings / sum_mask)[0]
        return embedding
