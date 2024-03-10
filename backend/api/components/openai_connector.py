from openai import OpenAI
from .utils.chat_history_formatter import ChatHistoryFormatter
from .utils.gpt_runner import GPTRunner
from .utils.prompt_builder import PromptBuilder
from .constants import performance_priorities, readability_priorities


class OpenaiConnector:
    def __init__(self, openai_api_key):
        self.client = OpenAI(api_key=openai_api_key)
        self.formatter = ChatHistoryFormatter()
        self.prompt_builder = PromptBuilder()
        self.runner = GPTRunner(self.client)

    def get_priority_statement_from_level(self, performance: int, readablity: int):
        performance_valid = performance in performance_priorities
        readability_valid = readablity in readability_priorities

        if performance_valid and readability_valid:
            return f"{performance_priorities[performance]} {readability_priorities[readablity]}"

        error = f"Invalid optimization level. The value can be 1, 2, or 3."
        raise ValueError(error)

    def create_newchat(self, code, version: str, performance: int, readablity: int):
        priority = self.get_priority_statement_from_level(performance, readablity)
        messages = self.prompt_builder.build_newchat_messages(code, version, priority)
        try:
            res = self.runner.get_gpt_response_from_messages(code, messages)
            return res
        except Exception as e:
            raise ValueError(f"Error requesting GPT response: {str(e)}")
