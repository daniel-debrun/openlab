"""The one error a study raises: what went wrong, what it means, and what to do."""


class StudyError(Exception):
    def __init__(self, error: str, meaning: str = "", fix: str = ""):
        super().__init__(error)
        self.error, self.meaning, self.fix = error, meaning, fix

    def __str__(self) -> str:
        return " -- ".join(x for x in (self.error, self.meaning, self.fix) if x)
