# llm_jssp

LLM-based solver of the Job Shop Scheduling Problem

## Installation

Requires `python3.8+`

To install with pip, run:

```
pip install git+https://github.com/andreimr/llm_jssp
```

## Usage

```
llm_jssp --help
```

### Tests

```bash
git clone 'https://github.com/andreimr/llm_jssp'
cd ./llm_jssp
pip install '.[testing]'
pytest
flake8 ./llm_jssp
mypy ./llm_jssp
```
