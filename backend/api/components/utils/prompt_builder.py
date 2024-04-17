from .file_handler import FileHandler
from ..constants import prompt_paths
import json


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

    def build_newchat_user_prompt(
        self, code: str, version: str, priority: str, gpu
    ) -> str:
        user_prompt = prompt_paths["new_task"].user
        user_prompt = self.file_handler.read_file(user_prompt)

        print("gpu::", gpu)

        gpu_string = self.build_gpu_config(gpu)

        return user_prompt.format(
            code=code, version=version, priority=priority, gpu=gpu_string
        )

    def build_newchat_system_prompt(self) -> str:
        return self.build_system_prompt("new_task")

    def build_gpu_config(self, gpu):
        config_db = self.file_handler.read_file("./gpu_specs.json")
        print("config_db::", config_db)

        gpu_config = json.loads(config_db)[gpu]
        print("gpu_config::", gpu_config)

        compute_capability = gpu_config["computeCapability"]
        memory_size = gpu_config["memorySizeGb"]
        bandwidth = gpu_config["memoryBandwidthGbPerSec"]
        supported_types = ", ".join(gpu_config["quantizationLevels"]["supportedTypes"])
        optimal_type = gpu_config["quantizationLevels"]["optimalType"]

        # Constructing the summary string
        summary = f"GPU Type: {gpu}. Compute Capability: {compute_capability}, Memory Size: {memory_size} GB, "
        summary += f"Memory Bandwidth: {bandwidth} GB/s, Supported Quantization Types: {supported_types}, "
        summary += f"Optimal Quantization Type: {optimal_type}"

        return summary

    def build_newchat_messages(self, code: str, version: str, priority: str, gpu):
        system_prompt = self.build_newchat_system_prompt()
        user_prompt = self.build_newchat_user_prompt(code, version, priority, gpu)

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


if __name__ == "__main__":
    prompt_builder = PromptBuilder()

    print(prompt_builder.build_gpu_config("TitanX"))
