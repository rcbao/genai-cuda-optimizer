from collections import namedtuple

PromptPath = namedtuple("PromptPath", ["system", "user"])

prompt_paths = {
    "new_task": PromptPath(
        system="../llm_prompts/openai_cuda_optimize_sys.txt",
        user="../llm_prompts/openai_cuda_optimize_user.txt",
    )
}

optimization_priorities = {
    1: "Improve code readability and maintainability primarily, with performance as a secondary concern.",
    2: "Balance execution speed with maintainability, ensuring reasonable code readability without sacrificing significant performance.",
    3: "Your task is to maximize the execution speed of the CUDA kernel code. You must completely disregard resource usage and code complexity.",
}

OPENAI_MODEL = "gpt-4-turbo-preview"
OPENAI_MAX_TOKENS = 2048
