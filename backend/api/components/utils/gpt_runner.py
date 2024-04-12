import re
import os
from openai import OpenAI
from dotenv import load_dotenv

from .file_handler import FileHandler
from .chat_history_formatter import ChatHistoryFormatter
from .prompt_builder import PromptBuilder
from ..constants import OPENAI_MODEL, OPENAI_MAX_TOKENS
from ..data_structures import OpenAIMessage

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")


class GPTRunner:
    def __init__(self, model=OPENAI_MODEL, max_tokens=OPENAI_MAX_TOKENS):
        self.client = OpenAI(api_key=openai_api_key)
        self.file_handler = FileHandler()
        self.formatter = ChatHistoryFormatter()
        self.prompt_builder = PromptBuilder()
        self.model = model
        self.max_tokens = max_tokens

    def get_gpt_response(self, messages: list, response_format=None) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format=response_format,
            max_tokens=self.max_tokens,
            temperature=0,
        )
        gpt_response = response.choices[0].message.content
        return gpt_response

    def get_cuda_kernel_signatures(self, text) -> list:
        """Extract CUDA kernel signatures from a given text."""
        pattern = r"__global__\s+void\s+[a-zA-Z_]\w*\s*\([^)]*\)"
        return re.findall(pattern, text)

    def get_shared_cuda_kernel_signatures(self, codes: list) -> list:
        signatures = []
        for code in codes:
            signatures += self.get_cuda_kernel_signatures(code)
        return list(set(signatures))

    def get_cuda_code_from_markdown(self, markdown_str: str) -> str:
        # Regular expression to find code block marked with ```cuda
        pattern = r"```cuda\n(.*?)```"
        match = re.search(pattern, markdown_str, re.DOTALL)
        if match:
            res = match.group(1)
            print("res: ", res)
            return res
        else:
            return None

    def swap_cuda_kernel(self, original: str, optimized: str, signature: str) -> str:
        """
        Replace the CUDA kernel in the original code with an optimized version based on the given signature.
        """
        # Escaping the signature for use in a regular expression
        escaped_signature = re.escape(signature)

        # Finding the exact match of the function signature in the original code
        match = re.search(escaped_signature, original)
        if not match:
            raise ValueError("Function signature not found in the original program.")

        start_index = match.start()

        # Find the matching pair of braces that enclose the kernel function
        brace_count = 0
        end_index = None
        for i in range(start_index, len(original)):
            if original[i] == "{":
                brace_count += 1
            elif original[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    end_index = i + 1  # Include the closing brace
                    break

        if end_index is None:
            raise ValueError("Could not find the end of the function body.")

        # Constructing the new program by replacing the original kernel with the optimized one
        new_program = original[:start_index] + optimized + original[end_index:]

        return new_program

    def get_gpt_response_from_messages(self, messages: list) -> dict[str, str]:
        response = self.get_gpt_response(messages)
        print("get_gpt_response_from_messages response:: ", response)

        if response:
            message = OpenAIMessage("assistant", response)
            return vars(message)
        else:
            raise ValueError("No CUDA code response from OpenAI model.")
