from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    gemini_api_key: SecretStr
    gemini_model: str = "gemini-2.5-flash"
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    chroma_collection_chapter: str = "harry_potter_chapterwise"
    chroma_collection_chunks: str = "harry_potter_chapterwise_chunks"
    n_retrieve: int = Field(default=25, gt=0)
    n_rerank: int = Field(default=5, gt=0)
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=100, gt=0)
    use_reranker: bool = True

    @model_validator(mode="after")
    def check_rerank(self):
        if self.n_rerank > self.n_retrieve:
            raise ValueError(
                f"reranked chunks ({self.n_rerank}) cannot exceed "
                f"retrieved chunks ({self.n_retrieve})"
            )
        return self
