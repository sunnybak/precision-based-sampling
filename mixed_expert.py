"""
Precision-Based Rating System with Mixed Experts

This module implements a sequential sampling algorithm for rating evaluation that combines
both human (LLM) and simulated expert ratings. It uses a precision-based stopping criterion
to determine the optimal number of ratings needed to achieve a desired confidence level.

The system supports:
- Multiple LLM models for evaluation
- Configurable Likert scale parameters
- Meta-prompt generation for evaluation diversity
- Both simulation and real evaluation modes

The confidence evaluation uses a sequential sampling approach that dynamically determines
the number of samples needed based on the observed variance in ratings.
"""

import numpy as np
import scipy.stats as st
import asyncio
from scipy.optimize import minimize
from litellm import acompletion
import konfigure
import matplotlib.pyplot as plt

import math
from typing import Callable, List, Dict, Optional, Union
import os
from itertools import cycle
import re

# Configuration Setup
# ------------------
path = os.path.join(os.path.dirname(__file__), 'config.yaml')
config = konfigure.load(path)

# Experiment Configuration
# Select which experiment configuration to use from config.yaml
EXPERIMENT = config.experiment_4

# Ground Truth Parameters (for simulation)
TRUE_MEAN = 8.3
TRUE_STD = 1

# Likert Scale Parameters
LIKERT_SCALE_MIN = 1
LIKERT_SCALE_MAX = 10
R = LIKERT_SCALE_MAX - LIKERT_SCALE_MIN  # Range of the scale
K = 10  # Precision parameter - higher values require more precision
ALPHA = 0.05  # Significance level for confidence intervals

# Hold the likert probability distribution
likert_probs = None

# Create a lock for thread-safe cycling
cycle_lock = asyncio.Lock()
model_cycle = cycle(EXPERIMENT.models)
prompt_cycle = None

def get_likert_probs(mean: float = TRUE_MEAN, 
                     std: float = TRUE_STD, 
                     scale_min: int = LIKERT_SCALE_MIN, 
                     scale_max: int = LIKERT_SCALE_MAX) -> np.ndarray:
    """Generate discrete probabilities for a Likert scale matching target parameters.
    
    Uses optimization to find a discrete probability distribution that matches
    the desired mean and standard deviation while maintaining valid probability
    constraints (sum to 1, all values between 0 and 1).
    
    Args:
        mean: Target mean for the distribution
        std: Target standard deviation
        scale_min: Minimum value on the Likert scale
        scale_max: Maximum value on the Likert scale
        
    Returns:
        np.ndarray of probabilities for each Likert scale value
    """
    global likert_probs
    # Only use cache if we're using the default parameters
    if (mean == TRUE_MEAN and std == TRUE_STD and 
        scale_min == LIKERT_SCALE_MIN and scale_max == LIKERT_SCALE_MAX and 
        likert_probs is not None):
        return likert_probs
    
    scale_size = scale_max - scale_min + 1
    probs = np.ones(scale_size) / scale_size
    
    def objective(probs):
        # Calculate current distribution parameters
        x = np.arange(scale_min, scale_max + 1)
        current_mean = np.sum(x * probs)
        current_std = np.sqrt(np.sum((x - current_mean)**2 * probs))
        
        # Minimize squared error from target parameters
        return (current_mean - mean)**2 + (current_std - std)**2
    
    # Optimization constraints and bounds
    constraints = ({'type': 'eq', 'fun': lambda p: np.sum(p) - 1})  # probabilities sum to 1
    bounds = [(0, 1) for _ in range(scale_size)]  # probabilities between 0 and 1
    
    # Find optimal probability distribution
    result = minimize(objective, probs, method='SLSQP', bounds=bounds, constraints=constraints)
    computed_probs = np.array(result.x)

    # Only cache if using default parameters
    if (mean == TRUE_MEAN and std == TRUE_STD and 
        scale_min == LIKERT_SCALE_MIN and scale_max == LIKERT_SCALE_MAX):
        likert_probs = computed_probs
        # Print distribution for debugging/verification
        prob_map = {i: f"{p:.4f}" for i, p in enumerate(computed_probs, start=LIKERT_SCALE_MIN)}
        print(f"Likert probability distribution for mean={mean:.3f} and std={std:.3f}:")
        for value, prob in prob_map.items():
            print(f"  {value}: {prob}")
        print()
    
    return computed_probs

def simulate_likert(response: str) -> float:
    """Sample a single Likert scale rating from the configured distribution.
    
    Args:
        response: Unused, included for API compatibility with LLM evaluator
        
    Returns:
        A single sampled Likert scale rating
    """
    probs = get_likert_probs()
    return np.random.choice(np.arange(LIKERT_SCALE_MIN, LIKERT_SCALE_MAX + 1), size=1, p=probs)[0]

async def init_prompts() -> None:
    """Initialize the evaluation prompts, either from config or by generation.
    
    If meta-prompts are enabled, generates multiple evaluation prompts using
    an LLM to increase diversity in the evaluation process.
    """
    global prompt_cycle
    # generate the prompts
    if EXPERIMENT.use_meta_prompt:
        prompts = await generate_eval_prompts()
        EXPERIMENT.generated_prompts = prompts # converts them to jinja templates
        eval_prompts = EXPERIMENT.generated_prompts
        print(f"Generated {len(eval_prompts)} evaluation prompts")
        # for p in eval_prompts:
        #     print(p, '\n\n')
    else:
        eval_prompts = [EXPERIMENT.prompt]
    prompt_cycle = cycle(eval_prompts)

async def generate_eval_prompts(n: int = 5) -> List[str]:
    """Generate evaluation prompts using a meta-prompt approach.
    
    Uses GPT-4 to generate diverse evaluation prompts that maintain
    consistent rating criteria while varying the presentation.
    
    Args:
        n: Number of prompts to generate
        
    Returns:
        List of generated evaluation prompts
    """
    meta_prompt = config.experiment_5.prompt.render(
        LIKERT_SCALE_MIN=LIKERT_SCALE_MIN,
        LIKERT_SCALE_MAX=LIKERT_SCALE_MAX,
        response="{{response}}", # to escape properly in Jinja template
    )

    async def generate_single_prompt():

        response = await acompletion(
            model='openai/gpt-4.1',
            messages=[{"role": "user", "content": meta_prompt}],
            temperature=1.0
        )
        return response.choices[0].message.content
    
    tasks = [generate_single_prompt() for _ in range(n)]
    return await asyncio.gather(*tasks)

async def get_llm_response(response: str) -> Optional[int]:
    """Get a Likert scale rating from an LLM evaluator.
    
    Sends the response to an LLM with a prompt requesting a rating,
    then extracts and validates the numerical rating from the response.
    
    Args:
        response: The text to be evaluated
        
    Returns:
        Integer rating or None if evaluation fails
    """
    rating = None
    try:
        # Get next model in a thread-safe way
        async with cycle_lock:
            selected_model = next(model_cycle)
            selected_prompt = next(prompt_cycle)

        response = await acompletion(
            model=selected_model,
            messages=[{
                "role": "user",
                "content": selected_prompt.render(
                    response=response,
                    LIKERT_SCALE_MIN=LIKERT_SCALE_MIN,
                    LIKERT_SCALE_MAX=LIKERT_SCALE_MAX,
                )
            }],
            # temperature=np.random.uniform(0.5, 1.0),
        )
        content = response.choices[0].message.content
        
        # Extract the first integer found
        match = re.search(r'\d+', content)
        if match:
            rating = int(match.group(0))
            if rating < LIKERT_SCALE_MIN or rating > LIKERT_SCALE_MAX:
                raise ValueError(f"Rating {rating} is out of range")
        else:
            raise ValueError(f"No valid rating found in response: {content}") 
    except Exception as e:
        print(f"Error getting LLM response: {e}")
        return None
    
    return rating

async def batch_eval(response: str, 
                    evaluator: Callable[[str], Union[float, int]], 
                    n: int) -> List[Union[float, int]]:
    """Run multiple evaluations in parallel.
    
    Args:
        response: Text to evaluate
        evaluator: Function that performs a single evaluation
        n: Number of evaluations to run
        
    Returns:
        List of successful evaluation results
    """
    async def run_evaluator(response: str):
        if asyncio.iscoroutinefunction(evaluator):
            return await evaluator(response)
        else:
            return evaluator(response)
    
    tasks = [run_evaluator(response) for _ in range(n)]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]

async def confidence_eval(
    batch_eval: Callable,
    evaluator: Callable[[str], Union[float, int]],
    alpha: float,
    K: int,
    R: float,
    response: str,
    n0: int = 5,
    verbose: bool = False,
) -> Dict[str, float]:
    """Sequential sampling algorithm with precision-based stopping criterion.
    
    Continues sampling until the confidence interval width meets the desired
    precision criterion: h ≤ R/(3K), where h is the confidence interval
    half-width.
    
    Args:
        batch_eval: Function to run parallel evaluations
        evaluator: Single evaluation function
        alpha: Significance level for confidence intervals
        K: Precision parameter (higher values require more precision)
        R: Range of the rating scale
        response: Text to evaluate
        n0: Initial number of samples
        verbose: Whether to print debug information
        
    Returns:
        Dict containing:
            - mean: Sample mean
            - h: Confidence interval half-width
            - n: Final number of samples
            - std: Sample standard deviation
    """
    z = st.norm.ppf(1 - alpha/2)  # z-score for the confidence interval
    x = list(await batch_eval(response, evaluator, n0))  # pilot samples
    if verbose: print(f"Pilot samples: {x}")
    
    # Target precision - half-width of desired confidence interval
    d = R / (3 * K)
    
    while True:
        n = len(x)  # current sample size
        std = np.std(x, ddof=1)  # sample standard deviation with Bessel's correction
        
        # Current confidence interval half-width
        h = z * std / np.sqrt(n)
        
        # Check if precision criterion is met
        if h <= d:
            # We have enough samples
            break
        
        # Calculate required sample size based on current variance
        delta = std / R  # normalized standard deviation
        n_req = math.ceil((3 * z * K * delta) ** 2)  # required sample size from power analysis
        
        # Get more samples, but limit batch size
        n_additional = max(1, min(n_req - n, n0))
        additional_samples = await batch_eval(response, evaluator, n_additional)
        if verbose: print(f"Additional samples: {additional_samples}")
        x.extend(additional_samples)
    
    return {
        'mean': np.mean(x),
        'h': h,
        'n': len(x),
        'std': std,
    }

async def run_evaluation(
        evaluator: Callable[[str], Union[float, int]] = simulate_likert,
        response: str = '',
        simulation: bool = False,
    ) -> None:
    """Run the complete evaluation process.
    
    Can operate in two modes:
    1. Regular evaluation: Uses the provided evaluator to rate the response
    2. Simulation: Runs multiple trials to validate the sampling algorithm
    
    Args:
        evaluator: Function to perform evaluations
        response: Text to evaluate
        simulation: Whether to run in simulation mode
    """
    z = st.norm.ppf(1 - ALPHA / 2)  # z-score for the confidence interval

    # Regular evaluation mode
    if not simulation:
        await init_prompts()
        result = await confidence_eval(
            batch_eval=batch_eval,
            evaluator=evaluator,
            alpha=ALPHA,
            K=K,
            R=R,
            response=response,
            verbose=not simulation,
        )
        print(f"Result:")
        print(f"  Rating: {result['mean']:.3f} ± {result['h']:.3f}")
        print(f"  Number of evaluator calls: {result['n']}")
        print(f"  Std: {result['std']:.3f}")

    # Simulation mode - validate algorithm properties
    if simulation:
        # Calculate theoretical sample size
        expected_n = math.ceil((3 * z * K * TRUE_STD / R) ** 2)
        print(f"Expected number of samples: {expected_n}")

        print("\n\nRunning 1000 simulation trials...")
        means, n_reqs, hs = [], [], []
        for _ in range(1000):
            result = await confidence_eval(
                batch_eval=batch_eval,
                evaluator=simulate_likert,
                alpha=ALPHA,
                K=K,
                R=R,
                response=response,
                verbose=not simulation)
            means.append(result['mean'])
            n_reqs.append(result['n'])
            hs.append(result['h'])
        
        # Report simulation results
        print(f"Grand Mean (1000 trials): {np.mean(means):.3f} ± {np.mean(hs):.3f}")
        print(f"True mean: {TRUE_MEAN}\n")
        print(f"Mean number of samples (1000 trials): {np.mean(n_reqs)}")
        print(f"Expected number of samples: {expected_n}")

async def plot_parameter_effects(classes=9, confidence=0.95, figsize=(18, 6)):
    """
    Plot how distribution parameters affect the expected sample size using existing simulation functions.
    
    Parameters:
    -----------
    classes : int
        Number of classes in the scale (default: 9)
    confidence : float
        Confidence level (default: 0.95)
    figsize : tuple
        Figure size for the plots (default: (18, 6))
    """
    alpha = 1 - confidence
    z = st.norm.ppf(1 - alpha / 2)
    
    # Helper function to create a custom simulator with specific parameters
    def create_custom_simulator(mean, std):
        def custom_simulate_likert(response: str) -> float:
            probs = get_likert_probs(mean, std, LIKERT_SCALE_MIN, LIKERT_SCALE_MAX)
            return np.random.choice(np.arange(LIKERT_SCALE_MIN, LIKERT_SCALE_MAX + 1), size=1, p=probs)[0]
        return custom_simulate_likert
    
    # Helper function to calculate theoretical expected sample size
    def expected_sample_size(std, K, alpha):
        z_val = st.norm.ppf(1 - alpha / 2)
        d = R / (3 * K)  # precision target
        return math.ceil((z_val * std / d) ** 2)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot 1: Std vs Sample Size
    plt.subplot(1, 3, 1)
    
    # Range of standard deviations to test
    stds_range = np.linspace(0.1, 2.0, 20)  # Reduced points for faster computation
    sample_sizes = []
    
    print("Computing effect of standard deviation on sample size...")
    for i, std in enumerate(stds_range):
        print(f"Progress: {i+1}/{len(stds_range)} (std={std:.2f})")
        
        # Create custom simulator with this std
        custom_simulator = create_custom_simulator(TRUE_MEAN, std)
        
        # Run multiple simulations to get average sample size
        n_trials = 10  # Reduced for faster computation
        trial_sizes = []
        
        for _ in range(n_trials):
            result = await confidence_eval(
                batch_eval=batch_eval,
                evaluator=custom_simulator,
                alpha=alpha,
                K=classes,
                R=R,
                response="",  # Not used in simulation
                verbose=False
            )
            trial_sizes.append(result['n'])
        
        sample_sizes.append(np.mean(trial_sizes))
    
    # Calculate normalized standard deviations and theoretical values
    normalized_stds = stds_range / R  # This is delta = std/R used in the algorithm
    theoretical_sizes = [expected_sample_size(std, classes, alpha) for std in stds_range]
    
    # Plot simulation results and theoretical line
    plt.plot(normalized_stds, sample_sizes, 'b-', label='Simulation Results', linewidth=2)
    plt.plot(normalized_stds, theoretical_sizes, 'r--', label='Expected No. Samples $n$', linewidth=2)
    
    plt.title(f'Effect of Normalized Standard Deviation\n(classes K = {classes}, confidence = {confidence})')
    plt.xlabel('Normalized Standard Deviation (δ = σ/R)')
    plt.ylabel('Expected Sample Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Effect of Confidence Intervals on Sample Size
    plt.subplot(1, 3, 2)
    
    # Range of confidence levels
    confidence_range = np.linspace(0.75, 0.99, 10)  # Reduced points for speed
    confidence_sample_sizes = []
    
    print("\nComputing effect of confidence level on sample size...")
    for i, conf in enumerate(confidence_range):
        print(f"Progress: {i+1}/{len(confidence_range)} (confidence={conf:.3f})")
        
        alpha_val = 1 - conf
        n_trials = 30  # Reduced for faster computation
        trial_sizes = []
        
        for _ in range(n_trials):
            result = await confidence_eval(
                batch_eval=batch_eval,
                evaluator=simulate_likert,  # Use default simulator
                alpha=alpha_val,
                K=classes,
                R=R,
                response="",
                verbose=False
            )
            trial_sizes.append(result['n'])
        
        confidence_sample_sizes.append(np.mean(trial_sizes))
    
    # Calculate theoretical values for confidence levels
    theoretical_conf_sizes = [expected_sample_size(TRUE_STD, classes, 1-conf) for conf in confidence_range]
    
    # Plot simulation results and theoretical line
    plt.plot(confidence_range, confidence_sample_sizes, 'b-', label='Simulation Results', linewidth=2)
    plt.plot(confidence_range, theoretical_conf_sizes, 'r--', label='Expected No. Samples $n$', linewidth=2)
    
    plt.title(f'Effect of Confidence Level\n(classes K = {classes}, std = {TRUE_STD})')
    plt.xlabel('Confidence Level')
    plt.ylabel('Expected Sample Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add dotted vertical lines for common confidence levels
    for conf in [0.8, 0.9, 0.95]:
        if conf >= confidence_range.min() and conf <= confidence_range.max():
            plt.axvline(x=conf, color='gray', linestyle=':', alpha=0.8, linewidth=2)
            # Find the corresponding y value for labeling
            idx = np.argmin(np.abs(confidence_range - conf))
            y_val = confidence_sample_sizes[idx]
            plt.text(conf+0.005, y_val+1, f"{conf:.2f}", fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Plot 3: Effect of number of classes K on Sample Size
    plt.subplot(1, 3, 3)
    
    # Range of classes
    class_range = np.arange(2, 16, 2)  # Reduced range for faster computation
    class_sample_sizes = []
    
    print("\nComputing effect of number of classes on sample size...")
    for i, k in enumerate(class_range):
        print(f"Progress: {i+1}/{len(class_range)} (classes={k})")
        
        n_trials = 30  # Reduced for faster computation
        trial_sizes = []
        
        for _ in range(n_trials):
            result = await confidence_eval(
                batch_eval=batch_eval,
                evaluator=simulate_likert,  # Use default simulator
                alpha=alpha,
                K=k,
                R=R,
                response="",
                verbose=False
            )
            trial_sizes.append(result['n'])
        
        class_sample_sizes.append(np.mean(trial_sizes))
    
    # Calculate theoretical values for different K values
    theoretical_class_sizes = [expected_sample_size(TRUE_STD, k, alpha) for k in class_range]
    
    # Plot simulation results and theoretical line
    plt.plot(class_range, class_sample_sizes, 'b-o', label='Simulation Results', linewidth=2, markersize=6)
    plt.plot(class_range, theoretical_class_sizes, 'r--s', label='Expected No. Samples $n$', linewidth=2, markersize=6)
    
    plt.title(f'Effect of Number of Classes K\n(confidence = {confidence}, std = {TRUE_STD})')
    plt.xlabel('Number of Classes K')
    plt.ylabel('Expected Sample Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(class_range)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(os.path.dirname(__file__), 'parameter_effects.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nPlots saved to '{output_path}'")

if __name__ == "__main__":
    # You can run either the original evaluation or the parameter effects plotting
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        asyncio.run(plot_parameter_effects())
    else:
        asyncio.run(run_evaluation(
            evaluator=get_llm_response,
            response=EXPERIMENT.response,
            simulation=True,
        ))
