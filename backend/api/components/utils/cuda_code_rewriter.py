import re
from .gpt_runner import GPTRunner
from .prompt_builder import PromptBuilder


class MarkdownCudaCodeParser:
    def __init__(self, markdown_text: str):
        self.code = self.extract_cuda_code_from_markdown_text(markdown_text)

    def extract_cuda_code_from_markdown_text(self, markdown_text: str) -> str:
        # Extract CUDA code from the markdown text
        pattern = r"```cuda\n(.*?)```"
        match = re.search(pattern, markdown_text, re.DOTALL)

        return match.group(1) if match else None


class SignatureDiff:
    def __init__(self, original_code: str, optimized_code: str):
        self.original_code = original_code
        self.optimized_code = optimized_code

        self.results = self.diff_kernel_signatures_from_code()

    def extract_kernel_signatures_from_code(self, code: str) -> set[str]:
        # Extract CUDA kernel signatures from a given text
        cuda_pattern = r"__global__\s+void\s+[a-zA-Z_]\w*\s*\([^)]*\)"
        results = re.findall(cuda_pattern, code)
        return set(results)

    def diff_kernel_signatures(self, original: list, optimized: list) -> dict:
        # Find the shared and new kernel signatures
        shared = original.intersection(optimized)
        new = optimized.difference(original)

        return {"shared": list(shared), "new": list(new)}

    def diff_kernel_signatures_from_code(self) -> dict[str, list[str]]:
        originals = self.extract_kernel_signatures_from_code(self.original_code)
        news = self.extract_kernel_signatures_from_code(self.optimized_code)

        return self.diff_kernel_signatures(originals, news)


class CudaImportAndGlobalMerger:
    def find_kernel_start(self, code: str) -> int:
        """Find the start index of the first CUDA kernel in the code."""
        pattern = r"__global__\s+void\s+[a-zA-Z_]\w*\s*\([^)]*\)"
        match = re.search(pattern, code)
        return match.start() if match else len(code)

    def extract_prekernel(self, code: str) -> str:
        """Extract the imports and global variable declarations from the code."""
        return code[: self.find_kernel_start(code)].strip()

    def merge_prekernels(self, code1: str, code2: str) -> str:
        """Merge two sections of code while removing duplicates."""
        combined_lines = set(code1.splitlines()) | set(code2.splitlines())
        return "\n".join(sorted(combined_lines)).strip()

    def add_new_imports_and_globals(self, original: str, optimized: str) -> str:
        """Replace the imports and globals in original_code with merged versions from both codes."""
        original_res = self.extract_prekernel(original)
        optimized_res = self.extract_prekernel(optimized)
        merged = self.merge_prekernels(original_res, optimized_res)

        # Append the kernel code from the original source, ensuring proper formatting.
        kernel_code_start = self.find_kernel_start(original)

        if merged:
            return f"{merged}\n{original[kernel_code_start:].strip()}"
        return original[kernel_code_start:].strip()


class CudaCodeRewriter:
    """
    A class to rewrite CUDA code based on the output of the OpenAI language model.

    It performs the following task in the steps below:
    1. Extract CUDA code from the markdown text
    2. Get the kernel signatures from the original and optimized code
    3. Swap the CUDA kernels in the original code with the optimized ones based on the shared signatures
    4. For any new signatures, add the optimized code to the response
    5. If the optimized code contains any new imports or global variables, add them to the the top of the response
    """

    def __init__(self, original_code: str, llm_response: str):
        self.original = original_code

        self.optimized = MarkdownCudaCodeParser(llm_response).code
        self.kernel_diff = SignatureDiff(original_code, self.optimized).results

        # set the rewritten code to the original code initially
        self.result = original_code
        self.runner = GPTRunner()
        self.prompt_builder = PromptBuilder()

    def integrate_shared_cuda_kernels(self):
        orig, optim = self.original, self.optimized
        messages = self.prompt_builder.build_rewrite_messages(orig, optim)
        try:
            return self.runner.get_gpt_response_from_messages(messages)
        except Exception as e:
            raise ValueError(f"Error requesting GPT response: {str(e)}")

    def rewrite(self) -> str:
        self.integrate_shared_cuda_kernels()
        return self.result
