import numpy as np

from tqdm import tqdm
from typing import List
from sklearn.preprocessing import normalize
from transformers.pipelines import Pipeline

from bertopic.backend import BaseEmbedder


class MeanPoolingEmbedder(BaseEmbedder):
    def __init__(self, embedding_model: Pipeline):
        super().__init__()

        if isinstance(embedding_model, Pipeline):
            self.embedding_model = embedding_model
        else:
            raise ValueError("Please select a correct transformers pipeline. For example: "
                             "pipeline('feature-extraction', model='distilbert-base-cased', device=0)")

    def embed(self,
              documents: List[str],
              max_tokens=512,
              verbose: bool = False) -> np.ndarray:
        """ Embed a list of n documents/words into an n-dimensional
        matrix of embeddings

        Arguments:
            documents: A list of documents or words to be embedded
            maximum_tokens: Maximum number of tokens per chunk
            verbose: Controls the verbosity of the process

        Returns:
            Document/words embeddings with shape (n, m) with `n` documents/words
            that each have an embeddings size of `m`
        """
        max_tokens = 512
        embeddings = []
        for document in tqdm(documents, total=len(documents), disable=not verbose):
            chunks = self._split_document(document, max_tokens)
            chunk_embeddings = [self._embed(chunk) for chunk in chunks]
            embeddings.append(np.mean(chunk_embeddings, axis=0))

        return np.array(embeddings)

    def _split_document(self, document: str, max_tokens: int) -> List[str]:
        tokens = self.embedding_model.tokenizer(document, return_tensors="pt")["input_ids"]
        chunks = [tokens[:, i:i + max_tokens - 7] for i in range(0, tokens.size(1), max_tokens - 7)]
        return [self.embedding_model.tokenizer.decode(chunk[0].tolist(), skip_special_tokens=True) for chunk in chunks]

    def _embed(self, chunk) -> np.ndarray:
        """ Mean pooling

        Arguments:
            chunk: The document chunk for which to extract the attention mask
        """
        features = self.embedding_model(chunk, truncation=True, padding=True)
        token_embeddings = np.array(features)
        attention_mask = self.embedding_model.tokenizer(chunk, truncation=True, padding=True, return_tensors="np")[
            "attention_mask"]
        input_mask_expanded = np.broadcast_to(np.expand_dims(attention_mask, -1), token_embeddings.shape)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = np.clip(input_mask_expanded.sum(1), a_min=1e-9, a_max=input_mask_expanded.sum(1).max())
        embedding = normalize(sum_embeddings / sum_mask)[0]
        return embedding
