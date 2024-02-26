class OpenAIMessage(dict):
    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content

    def __getattr__(self, attr):
        return self[attr]
