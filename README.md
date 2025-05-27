# Precision-Based Rating System

[Precision-based sampling](https://sunnybak.net/blog/precision-based-sampling)

## Overview

This module implements a sequential sampling algorithm for rating evaluation that combines both human (LLM) and simulated expert ratings. It uses a precision-based stopping criterion to determine the optimal number of ratings needed to achieve a desired confidence level.

## Features

- **Sequential Sampling**: Dynamically determines the number of samples needed based on observed variance
- **Multiple LLM Models**: Supports evaluation using different LLM models
- **Configurable Likert Scale**: Customizable scale parameters and precision requirements
- **Meta-prompt Generation**: Generates diverse evaluation prompts for increased evaluation diversity
- **Simulation Mode**: Validates algorithm properties using simulated data
- **Parameter Analysis**: Visualizes how different parameters affect sample size requirements

## Installation

Ensure you have the required dependencies:

```bash
pip install numpy scipy matplotlib litellm konfigure asyncio
```

## Usage

### 1. Basic Evaluation

Run a standard evaluation with LLM evaluators:

```bash
python mixed_expert.py
```

This will:
- Use the configured LLM models to evaluate the response
- Apply sequential sampling until precision target is met
- Output the final rating with confidence interval

### 2. Simulation Mode

Run in simulation mode to validate the algorithm:

```bash
python mixed_expert.py
```

Set `simulation=True` in the `run_evaluation()` call to:
- Run 1000 simulation trials
- Compare actual vs theoretical sample sizes
- Validate algorithm convergence properties

### 3. Parameter Effects Visualization

Generate plots showing how different parameters affect sample size:

```bash
python mixed_expert.py plot
```

This creates three plots showing:
- **Plot 1**: Effect of normalized standard deviation (δ = σ/R) on sample size
- **Plot 2**: Effect of confidence level on sample size (with reference lines at 0.8, 0.9, 0.95)
- **Plot 3**: Effect of number of classes (K) on sample size

The plots compare simulation results with theoretical predictions and are saved as `parameter_effects.png`.

## Configuration

### Key Parameters

Edit the configuration in `mixed_expert.py`:

```python
# Likert Scale Parameters
LIKERT_SCALE_MIN = 1
LIKERT_SCALE_MAX = 10
K = 10  # Precision parameter - higher values require more precision
ALPHA = 0.05  # Significance level for confidence intervals

# Ground Truth Parameters (for simulation)
TRUE_MEAN = 8.3
TRUE_STD = 1
```

### Experiment Configuration

The system uses configuration from `config.yaml`. Select which experiment to run:

```python
EXPERIMENT = config.experiment_4  # Change to desired experiment
```

### LLM Models

Configure which models to use in your `config.yaml`:

```yaml
experiment_4:
  models:
    - "openai/gpt-4"
    - "anthropic/claude-3-sonnet"
    - "openai/gpt-3.5-turbo"
```

## Algorithm Details

### Precision-Based Stopping Criterion

The algorithm continues sampling until the confidence interval half-width meets:

```
h ≤ R/(3K)
```

Where:
- `h` = confidence interval half-width
- `R` = range of the rating scale
- `K` = precision parameter

### Expected Sample Size Formula

The theoretical expected sample size is:

```
n = ceil((z * σ / d)²)
```

Where:
- `z` = z-score for the confidence level
- `σ` = standard deviation
- `d` = precision target = R/(3K)

### Sequential Sampling Process

1. **Pilot Phase**: Collect initial samples (default: 5)
2. **Evaluation**: Calculate current confidence interval half-width
3. **Decision**: Check if precision criterion is met
4. **Continuation**: If not, estimate required additional samples and continue
5. **Termination**: Stop when precision target is achieved

## Output

### Standard Evaluation
```
Result:
  Rating: 8.245 ± 0.312
  Number of evaluator calls: 23
  Std: 1.156
```

### Simulation Results
```
Expected number of samples: 25
Grand Mean (1000 trials): 8.301 ± 0.333
True mean: 8.3
Mean number of samples (1000 trials): 24.7
```

## Files

- `mixed_expert.py`: Main implementation
- `config.yaml`: Configuration file for experiments
- `parameter_effects.png`: Generated plots (when using plot mode)
- `README.md`: This documentation

## Customization

### Adding New Evaluators

Create custom evaluation functions:

```python
async def my_custom_evaluator(response: str) -> Optional[int]:
    # Your evaluation logic here
    return rating  # Return integer between LIKERT_SCALE_MIN and LIKERT_SCALE_MAX
```

### Modifying Precision Requirements

Adjust the precision parameter `K`:
- Higher `K` = more precision required = more samples needed
- Lower `K` = less precision required = fewer samples needed

### Changing Confidence Levels

Modify `ALPHA` for different confidence levels:
- `ALPHA = 0.05` → 95% confidence
- `ALPHA = 0.10` → 90% confidence
- `ALPHA = 0.01` → 99% confidence