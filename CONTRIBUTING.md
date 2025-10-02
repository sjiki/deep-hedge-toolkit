# Contributing to Deep Hedge Strategy Toolkit

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on GitHub with:
- Clear description of the bug
- Steps to reproduce
- Expected behavior vs. actual behavior
- Python version and operating system
- Any relevant error messages or screenshots

### Suggesting Enhancements

We welcome suggestions for new features or improvements:
- Open an issue describing your suggestion
- Explain the use case and benefits
- Provide examples if applicable

### Pull Requests

1. **Fork the repository**
   ```bash
   git clone https://github.com/sjiki/deep-hedge-toolkit.git
   cd deep-hedge-toolkit
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the existing code style
   - Add comments for complex logic
   - Update documentation as needed

4. **Test your changes**
   - Ensure existing functionality still works
   - Test edge cases
   - Verify calculations are accurate

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add feature: brief description"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request**
   - Provide a clear description of changes
   - Reference any related issues
   - Explain testing performed

## Code Style Guidelines

### Python Code
- Follow PEP 8 style guide
- Use meaningful variable and function names
- Add docstrings for functions and classes
- Keep functions focused and modular
- Use type hints where appropriate

Example:
```python
def calculate_hedge_ratio(portfolio_value: float, beta: float,
                         index_price: float, multiplier: int = 100) -> float:
    """
    Calculate the hedge ratio for portfolio protection.

    Args:
        portfolio_value: Total portfolio value in dollars
        beta: Portfolio beta relative to index
        index_price: Current index price
        multiplier: Contract multiplier (default: 100)

    Returns:
        Number of contracts needed for hedge
    """
    return (portfolio_value * beta) / (index_price * multiplier)
```

### Documentation
- Use clear, concise language
- Provide examples for complex features
- Keep README.md up to date
- Update version history for significant changes

### Commits
- Use descriptive commit messages
- Start with a verb (Add, Fix, Update, Remove, etc.)
- Keep commits focused on a single change
- Reference issue numbers when applicable

Good commit messages:
```
✓ Add Monte Carlo simulation for volatility scenarios
✓ Fix calculation error in put spread pricing
✓ Update README with installation instructions
✓ Remove deprecated function from hedge calculator
```

Poor commit messages:
```
✗ Fixed stuff
✗ Updates
✗ Changes to code
```

## Areas for Contribution

We especially welcome contributions in these areas:

### 1. New Features
- Additional hedging strategies (iron condors, butterflies, etc.)
- Integration with real-time market data APIs
- Visualization tools (charts, graphs, dashboards)
- Portfolio rebalancing algorithms
- Tax optimization strategies

### 2. Enhancements
- Performance optimization
- Additional backtesting scenarios
- Enhanced error handling
- Improved user interface
- Better logging and debugging

### 3. Documentation
- Tutorial videos or guides
- Case studies with real examples
- Translation to other languages
- API documentation improvements
- FAQ section

### 4. Testing
- Unit tests for calculation functions
- Integration tests for workflow
- Edge case testing
- Performance benchmarks

### 5. Tools & Utilities
- Data import/export utilities
- Broker integration scripts
- Automated reporting tools
- Portfolio analysis helpers

## Financial Calculations

When contributing code that involves financial calculations:

1. **Accuracy is critical**
   - Double-check formulas against established sources
   - Test with known values and edge cases
   - Document assumptions clearly

2. **Include references**
   - Cite sources for formulas (e.g., Black-Scholes model)
   - Link to academic papers or textbooks
   - Explain any simplifications made

3. **Consider edge cases**
   - Zero or negative values
   - Very large or very small numbers
   - Division by zero
   - Expiry date = current date

4. **Add tests**
   - Verify calculations match expected results
   - Test boundary conditions
   - Compare against established libraries (where applicable)

## Legal & Compliance

**Important:** This is an educational toolkit, NOT financial advice.

When contributing:
- Do not make claims about guaranteed returns
- Include appropriate disclaimers
- Emphasize the educational nature of the tool
- Avoid language that could be construed as investment advice
- Do not include real account credentials or personal trading data

## Questions?

If you have questions about contributing:
- Open an issue with the "question" label
- Review existing issues and pull requests
- Check the documentation in `deep_hedge_workflow.md`

## Code of Conduct

- Be respectful and professional
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Accept that disagreements happen
- Assume good intentions

## Recognition

Contributors will be recognized in:
- GitHub contributors list
- Release notes (for significant contributions)
- README acknowledgments section (future)

---

Thank you for helping improve the Deep Hedge Strategy Toolkit! Your contributions make this project better for everyone.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
