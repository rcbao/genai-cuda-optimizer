from .gpt_runner import GPTRunner
from .prompt_builder import PromptBuilder
from ..constants import LIGHTWEIGHT_OPENAI_MODEL


class CudaReasonGenerator:

    def __init__(self, optimized_code: str):

        self.optimized = optimized_code

        self.runner = GPTRunner(model=LIGHTWEIGHT_OPENAI_MODEL)
        self.prompt_builder = PromptBuilder()

    def generate(self):
        messages = self.prompt_builder.build_reasons_messages(self.optimized)
        try:
            message = self.runner.get_gpt_response_from_messages(messages)
            return message["content"]
        except Exception as e:
            raise ValueError(f"Error requesting GPT response: {str(e)}")
