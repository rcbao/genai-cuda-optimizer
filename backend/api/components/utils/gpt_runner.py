import re
import json
from llama_index.core.chat_engine import (
    CondensePlusContextChatEngine,
)
from llama_index.core import PromptTemplate
from .file_handler import FileHandler
from .chat_history_formatter import ChatHistoryFormatter
from .prompt_builder import PromptBuilder
from ..constants import prompt_paths, OPENAI_MODEL, OPENAI_MAX_TOKENS
from llama_index.core.chat_engine.types import AgentChatResponse
from ..data_structures import OpenAIMessage


SHOWING_ONLY_ONE_CONTEXT = True
RAG_DISABLED = False


class GPTRunner:
    def __init__(self, client):
        self.client = client
        self.file_handler = FileHandler()
        self.formatter = ChatHistoryFormatter()
        self.prompt_builder = PromptBuilder()

    def get_gpt_response(self, messages: list, response_format=None) -> str:
        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            response_format=response_format,
            max_tokens=OPENAI_MAX_TOKENS,
            temperature=0,
        )
        gpt_response = response.choices[0].message.content
        return gpt_response

    def get_gpt_response_from_messages(self, messages: list):
        response = self.get_gpt_response(messages)

        response = OpenAIMessage("assistant", response)

        return vars(response)
