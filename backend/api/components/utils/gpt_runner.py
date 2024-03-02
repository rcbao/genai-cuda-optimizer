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

    def extract_cuda_kernel_signatures(self, text) -> list:
        """Extract CUDA kernel signatures from a given text."""
        pattern = r"__global__\s+void\s+[a-zA-Z_]\w*\s*\([^)]*\)"
        return re.findall(pattern, text)

    def extract_shared_cuda_kernel_signatures(self, codes: list) -> list:
        signatures = []
        for code in codes:
            signatures += self.extract_cuda_kernel_signatures(code)
        return list(set(signatures))

    def extract_cuda_code_from_markdown(self, markdown_str: str) -> str:
        # Regular expression to find code block marked with ```cuda
        pattern = r"```cuda\n(.*?)```"
        match = re.search(pattern, markdown_str, re.DOTALL)
        if match:
            return match.group(1)
        else:
            return None

    def swap_cuda_kernel(self, original_code: str, optimized_code: str, signature: str):
        # Escape special characters in function signature for regex pattern
        pattern = re.escape(signature)

        # Check if the function signature is present in the original program
        if not re.search(pattern, original_code):
            raise ValueError("Function signature not found in the original program.")

        # Replace the original kernel function with the optimized one
        start_index = original_code.find(signature)
        end_index = (
            start_index + len(re.findall(pattern, original_code)[0][0]) + len(signature)
        )

        # Adjust end_index to include all up to the closing brace of the original function
        brace_count = 0
        for i in range(start_index, len(original_code)):
            if original_code[i] == "{":
                brace_count += 1
            elif original_code[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    end_index = i + 1  # Include the closing brace
                    break

        # Now construct the new program
        new_program = (
            original_code[:start_index] + optimized_code + original_code[end_index:]
        )

        return new_program

    def get_gpt_response_from_messages(self, original_code: str, messages: list):
        llm_response = self.get_gpt_response(messages)
        optimized_code = self.extract_cuda_code_from_markdown(llm_response)

        codes = [original_code, optimized_code]
        signatures = self.extract_shared_cuda_kernel_signatures(codes)

        response = ""

        for s in signatures:
            swap_source = response if response else optimized_code
            response = self.swap_cuda_kernel(original_code, swap_source, s)

        if response:
            response = OpenAIMessage("assistant", response)
            return vars(response)
        else:
            raise ValueError("No CUDA code response from OpenAI model.")
