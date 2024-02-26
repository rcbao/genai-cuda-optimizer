from collections import namedtuple

PromptPath = namedtuple("PromptPath", ["system", "user"])

prompt_paths = {
    "new_task": PromptPath(
        system="../llm_prompts/openai_cuda_optimize_sys.txt",
        user="../llm_prompts/openai_cuda_optimize_user.txt",
    )
}

OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_MAX_TOKENS = 300
