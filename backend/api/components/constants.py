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
    "reasons": PromptPath(
        system="../llm_prompts/openai_cuda_reasons_sys.txt",
        user="../llm_prompts/openai_cuda_reasons_user.txt",
    ),
}

performance_priorities = {
    1: "You MUST enhance the execution speed with minimal modifications. Prioritize straightforward and easily implementable optimizations.",
    2: "You MUST optimize the code for faster execution speed through effective strategies.",
    3: "You MUST optimize the code as much as possible using all available means, including advanced features.",
}


readability_priorities = {
    1: "You must also improve code readability with small changes.",
    2: "You must also improve readability by applying best practices.",
    3: "You must maximize code readability with comprehensive changes.",
}


OPENAI_MODEL = "gpt-4-turbo"
LIGHTWEIGHT_OPENAI_MODEL = "gpt-3.5-turbo"

OPENAI_MAX_TOKENS = 2048
