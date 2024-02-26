import re
from .file_handler import FileHandler
from .chat_history_formatter import ChatHistoryFormatter
from .prompt_builder import PromptBuilder
from ..constants import OPENAI_MODEL, OPENAI_MAX_TOKENS
from ..data_structures import OpenAIMessage


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

    def extract_cuda_code(self, markdown_str: str) -> str:
        # Regular expression to find code block marked with ```cuda
        pattern = r"```cuda\n(.*?)```"
        match = re.search(pattern, markdown_str, re.DOTALL)
        if match:
            return match.group(1)
        else:
            return None

    def get_gpt_response_from_messages(self, messages: list):
        response = self.get_gpt_response(messages)
        response = self.extract_cuda_code(response)

        if response:
            print(response)
            response = OpenAIMessage("assistant", response)
            return vars(response)
        else:
            raise ValueError("No CUDA code response from OpenAI model.")
