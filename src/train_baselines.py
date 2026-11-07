"""Compatibility wrapper. Prefer the task-specific module commands."""
from src.task1.train import main as task1
from src.task2.train import main as task2
from src.task3.train import main as task3

if __name__ == "__main__":
    print("Use: python -m src.task1.train")
    print("Use: python -m src.task2.train")
    print("Use: python -m src.task3.train")
    print("Use: python -m src.task5.train")
