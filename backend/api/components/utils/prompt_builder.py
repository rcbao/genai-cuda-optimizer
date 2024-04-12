from .file_handler import FileHandler
from ..constants import prompt_paths


class PromptBuilder:
    def __init__(self):
        self.file_handler = FileHandler()

    def build_system_prompt(self, prompt_type: str) -> str:
        system_prompt = prompt_paths[prompt_type].system
        return self.file_handler.read_file(system_prompt)

    def build_messages(self, system_prompt, user_prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return messages

    def build_newchat_user_prompt(self, code: str, version: str, priority: str) -> str:
        user_prompt = prompt_paths["new_task"].user
        user_prompt = self.file_handler.read_file(user_prompt)

        return user_prompt.format(code=code, version=version, priority=priority)

    def build_newchat_system_prompt(self) -> str:
        return self.build_system_prompt("new_task")

    def build_newchat_messages(self, code: str, version: str, priority: str):
        system_prompt = self.build_newchat_system_prompt()
        user_prompt = self.build_newchat_user_prompt(code, version, priority)

        return self.build_messages(system_prompt, user_prompt)

    def build_rewrite_user_prompt(
        self, original: str, optimized: str, signatures: dict[str, list[str]]
    ) -> str:
        user_prompt = prompt_paths["integrate_changes"].user
        user_prompt = self.file_handler.read_file(user_prompt)

        shared_signatures = ", ".join(signatures["shared"])
        new_signatures = ", ".join(signatures["new"])

        res = user_prompt.format(
            original=original,
            optimized=optimized,
            shared=shared_signatures,
            new=new_signatures,
        )
        return res

    def build_rewrite_system_prompt(self) -> str:
        return self.build_system_prompt("integrate_changes")

    def build_rewrite_messages(self, original_code, optimized_code, signatures):
        system_prompt = self.build_rewrite_system_prompt()
        user_prompt = self.build_rewrite_user_prompt(
            original_code, optimized_code, signatures
        )

        return self.build_messages(system_prompt, user_prompt)

    def build_reasons_user_prompt(self, optimized: str) -> str:
        user_prompt = prompt_paths["reasons"].user
        user_prompt = self.file_handler.read_file(user_prompt)

        return user_prompt.format(optimized=optimized)

    def build_reasons_system_prompt(self) -> str:
        return self.build_system_prompt("reasons")

    def build_reasons_messages(self, optimized_code):
        system_prompt = self.build_reasons_system_prompt()
        user_prompt = self.build_reasons_user_prompt(optimized_code)

        return self.build_messages(system_prompt, user_prompt)
