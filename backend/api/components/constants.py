from collections import namedtuple

PromptPath = namedtuple("PromptPath", ["system", "user"])

prompt_paths = {
    "new_task": PromptPath(
        system="../llm_prompts/openai_cuda_optimize_sys.txt",
        user="../llm_prompts/openai_cuda_optimize_user.txt",
    ),
    "integrate_changes": PromptPath(
        system="../llm_prompts/openai_cuda_integrate_sys.txt",
        user="../llm_prompts/openai_cuda_integrate_user.txt",
    ),
}

performance_priorities = {
    1: "Your task is to enhance the execution speed with minimal modifications. Prioritize straightforward and easily implementable optimizations.",
    2: "Your task is optimize the code for faster execution speed through effective strategies.",
    3: "Your goal is to achieve the highest possible acceleration of CUDA kernel execution. Use all available means to optimize performance.",
}


readability_priorities = {
    1: "You must improve code readability with minimal changes. Focus on clear and simple enhancements.",
    2: "You must enhance code readability by applying best practices. Streamline and clarify the code.",
    3: "You must maximize code readability. Apply comprehensive strategies to ensure the code is as clear and intuitive as possible.",
}


OPENAI_MODEL = "gpt-4-turbo-preview"
OPENAI_MAX_TOKENS = 2048
