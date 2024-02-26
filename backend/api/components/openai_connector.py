from openai import OpenAI
from .utils.chat_history_formatter import ChatHistoryFormatter
from .utils.gpt_runner import GPTRunner
from .utils.prompt_builder import PromptBuilder


class OpenaiConnector:
    def __init__(self, openai_api_key):
        self.client = OpenAI(api_key=openai_api_key)
        self.formatter = ChatHistoryFormatter()
        self.prompt_builder = PromptBuilder()
        self.runner = GPTRunner(self.client)

    def create_newchat(self, code: str, version: str, level: int):
        messages = self.prompt_builder.build_newchat_messages(code, version, level)
        try:
            res = self.runner.get_gpt_response_from_messages(messages)
            return res
        except Exception as e:
            raise ValueError(f"Error requesting GPT response: {str(e)}")
