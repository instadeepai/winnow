# Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire and create, and we welcome your support! Any contributions you make are **greatly appreciated**.

## Ways to contribute

If you have ideas for enhancements, you can:

- Fork the repository and submit a pull request
- Open an issue and tag it with "enhancement"
- Report bugs by opening an issue with the "bug" label
- Improve documentation
- Add examples or tutorials

## Contribution process

1. **Fork the repository**

    ```bash
    git clone https://github.com/instadeepai/winnow.git
    cd winnow
    ```
2. **Create a feature branch**

    ```bash
    git checkout -b feat-amazing-feature
    ```
3. **Set up development environment**

    ```bash
    # Install development dependencies
    uv sync --dev

    # Set up pre-commit hooks
    pre-commit install
    ```
4. **Make your changes**

    - Follow the existing code style
    - Add tests for new functionality
    - Update documentation as needed
5. **Test your changes**

    ```bash
    # Run tests
    pytest

    # Check code formatting
    pre-commit run --all-files
    ```
6. **Commit your changes**

    ```bash
    git commit -m 'feat: add some amazing feature'
    ```
7. **Push to your branch**

    ```bash
    git push origin feat-amazing-feature
    ```
8. **Open a Pull Request**

    - Provide a clear description of your changes
    - Reference any related issues
    - Include examples if applicable

## Development guidelines

### Code style

This project uses:

- **Ruff** for linting and formatting
- **Pre-commit** hooks for automated checks
- **Type hints** where applicable

### Testing

- Write tests for new functionality using `pytest`
- Ensure all tests pass before submitting a PR
- Include both unit tests and integration tests where appropriate

### Documentation

- Update docstrings for new or modified functions/classes
- Add examples to demonstrate usage
- Update the API documentation pages if you add new modules

### Issues and bug reports

When reporting bugs, please include:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected vs. actual behaviour
- Your environment details (Python version, OS, etc.)
- Any relevant error messages or logs

### Feature requests

For feature requests, please:

- Use a clear and descriptive title
- Explain the motivation for the feature
- Describe the proposed solution
- Consider alternatives you've thought about

## Getting help

If you need help or have questions:

- Check the [examples](examples.md) for usage patterns
- Look through existing [issues](https://github.com/instadeepai/winnow/issues)
- Open a new issue with the "question" label

## Recognition

Contributors are recognised in the project's contributor list. Don't forget to give the project a star! ⭐

Thank you for considering contributing to Winnow!
