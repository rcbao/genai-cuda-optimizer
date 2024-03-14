import re


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

    def extract_kernel_signatures_from_code(self, code: str) -> list[str]:
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
        self.kernel_diff = SignatureDiff(original_code, self.optimized).results()

        # set the rewritten code to the original code initially
        self.result = original_code

    def add_new_imports_and_globals(self) -> str:
        merger = CudaImportAndGlobalMerger()
        self.result = merger.add_new_imports_and_globals(self.result, self.optimized)

    def swap_shared_cuda_kernels(self):

        shared_kernels = self.kernel_diff["new"]

        for signature in shared_kernels:
            # Escape special characters in the signature for regex usage.
            escaped_signature = re.escape(signature)

            # Define regex pattern to find the kernel body.
            # This pattern assumes that the kernel body is enclosed in braces {}.
            pattern = rf"{escaped_signature}\s*\{{(.*?)\}}"

            # Find the kernel body in the optimized code.
            optimized_kernel_match = re.search(pattern, self.optimized, re.DOTALL)
            if optimized_kernel_match:
                optimized_kernel_body = optimized_kernel_match.group(1)

                # Replace the kernel body in the original code with the optimized one.
                # It assumes that the same kernel signature will not appear more than once in the original code.
                repl = rf"{signature} {{{optimized_kernel_body}}}"
                self.result = re.sub(pattern, repl, self.result, count=1)

    def add_new_cuda_kernels(self) -> str:
        new_kernels = []

        # Extract new kernels from the optimized code.
        for signature in self.kernel_diff["new"]:
            pattern = re.escape(signature) + r"\s*\{.*?\}\s*"
            match = re.search(pattern, self.optimized, re.DOTALL)
            if match:
                # Trim leading/trailing whitespace to ensure consistent formatting.
                new_kernels.append(match.group().strip())

        # Identify the insertion point after the last kernel in the original code.
        insertion_point = 0
        for match in re.finditer(r"\}\s*(?=\Z|\s*__global__)", self.result, re.DOTALL):
            insertion_point = match.end()

        # Combine the code segments, ensuring consistent newline usage.
        # Only add newlines if there are existing kernels; adjust for both original code and new kernels.
        if self.result and new_kernels:
            prev = self.result[:insertion_point].rstrip()
            nxt = self.result[insertion_point:].lstrip()
            updated_code = prev + "\n\n" + "\n\n".join(new_kernels) + "\n" + nxt
        elif not self.result and new_kernels:
            updated_code = "\n\n".join(new_kernels)
        else:
            updated_code = self.result

        self.result = updated_code.strip()

    def rewrite(self) -> str:
        self.add_new_imports_and_globals()
        self.swap_shared_cuda_kernels()
        self.add_new_cuda_kernels()
        return self.result
