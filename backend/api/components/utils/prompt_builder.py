from .file_handler import FileHandler
from ..constants import prompt_paths


class PromptBuilder:
    def __init__(self):
        self.file_handler = FileHandler()

    def build_newchat_user_prompt(self, code: str, version: str, priority: str) -> str:
        user_prompt = prompt_paths["new_task"].user
        user_prompt = self.file_handler.read_file(user_prompt)

        return user_prompt.format(code=code, version=version, priority=priority)

    def build_newchat_system_prompt(self) -> str:
        system_prompt = prompt_paths["new_task"].system
        return self.file_handler.read_file(system_prompt)

    def build_messages(self, system_prompt, user_prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return messages

    def build_newchat_messages(self, code: str, version: str, priority: str):
        system_prompt = self.build_newchat_system_prompt()
        user_prompt = self.build_newchat_user_prompt(code, version, priority)

        return self.build_messages(system_prompt, user_prompt)
