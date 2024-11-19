# Define variables for the tools
MYPY = mypy
PYFLAKES = pyflakes
RUFF = ruff

# Define the source directories
SRC_DIR = sound_spec
TEST_DIR = tests

# Target to run mypy type checks
.PHONY: mypy
mypy:
    $(MYPY) $(SRC_DIR) $(TEST_DIR)

# Target to run pyflakes linting
.PHONY: pyflakes
pyflakes:
    $(PYFLAKES) $(SRC_DIR) $(TEST_DIR)

# Target to run ruff formatting
.PHONY: ruff
ruff:
    $(RUFF) check $(SRC_DIR) $(TEST_DIR)

# Target to run all checks
.PHONY: all
all: mypy pyflakes ruff

# Target to automatically format code using ruff
.PHONY: format
format:
    $(RUFF) format $(SRC_DIR) $(TEST_DIR)
