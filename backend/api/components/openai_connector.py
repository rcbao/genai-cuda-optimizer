from openai import OpenAI
from .utils.chat_history_formatter import ChatHistoryFormatter
from .utils.gpt_runner import GPTRunner
from .utils.prompt_builder import PromptBuilder
from .constants import performance_priorities, readability_priorities
from .utils.cuda_code_rewriter import CudaCodeRewriter
from .utils.cuda_reason_generator import CudaReasonGenerator


class OpenaiConnector:
    def __init__(self, openai_api_key):
        self.client = OpenAI(api_key=openai_api_key)
        self.formatter = ChatHistoryFormatter()
        self.prompt_builder = PromptBuilder()
        self.runner = GPTRunner()

    def get_priority_statement_from_level(self, performance: int, readablity: int):
        print("get_priority_statement_from_level")
        performance_valid = int(performance) in performance_priorities
        readability_valid = int(readablity) in readability_priorities

        if performance_valid and readability_valid:
            print("get_priority_statement_from_level return")
            res = (
                performance_priorities[int(performance)]
                + " "
                + readability_priorities[int(readablity)]
            )
            print(res)
            return res
        print("error happend in get_priority_statement_from_level")
        error = f"Invalid optimization level. The value can be 1, 2, or 3."
        raise ValueError(error)

    def create_newchat(self, code, version: str, performance: int, readablity: int):
        priority = self.get_priority_statement_from_level(performance, readablity)
        messages = self.prompt_builder.build_newchat_messages(code, version, priority)
        try:
            llm_response = self.runner.get_gpt_response_from_messages(messages)
            rewriter = CudaCodeRewriter(code, llm_response)
            optimized_code = rewriter.rewrite().content

            generator = CudaReasonGenerator(optimized_code)
            reason_response = generator.generate_reasons().content

            return {
                "content": optimized_code,
                "reasons": reason_response,
            }

        except Exception as e:
            raise ValueError(f"Error requesting GPT response: {str(e)}")
