from openai import OpenAI
from .utils.chat_history_formatter import ChatHistoryFormatter
from .utils.gpt_runner import GPTRunner
from .utils.prompt_builder import PromptBuilder
from .constants import optimization_priorities


class OpenaiConnector:
    def __init__(self, openai_api_key):
        self.client = OpenAI(api_key=openai_api_key)
        self.formatter = ChatHistoryFormatter()
        self.prompt_builder = PromptBuilder()
        self.runner = GPTRunner(self.client)

    def get_priority_statement_from_level(self, level: int):
        if level in optimization_priorities:
            return optimization_priorities[level]

        error = f"Invalid optimization level: {level}. The value can be 1, 2, or 3."
        raise ValueError(error)

    def create_newchat(self, code: str, version: str, level: str):
        priority = self.get_priority_statement_from_level(level)
        messages = self.prompt_builder.build_newchat_messages(code, version, priority)
        try:
            res = self.runner.get_gpt_response_from_messages(code, messages)
            return res
        except Exception as e:
            raise ValueError(f"Error requesting GPT response: {str(e)}")
