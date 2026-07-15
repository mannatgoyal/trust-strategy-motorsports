# Contributing Guidelines

We welcome contributions to the **F1 Strategy Engineer Toolkit**!

## How to Contribute
1.  **Fork** the repository and clone it locally.
2.  Install development dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Implement changes following **SOLID** design principles and clean type annotations.
4.  Write comprehensive tests for any new modules inside the `tests/` directory.
5.  Execute the unit test suite before pushing:
    ```bash
    python -m unittest discover -s tests -p "test_*.py"
    ```
6.  Ensure the stream dashboard (`f1_ai_strategist.py`) runs successfully locally.
7.  Submit a **Pull Request** detailing the implementation, assumptions, and validation metrics.
