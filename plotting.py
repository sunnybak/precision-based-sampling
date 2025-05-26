import os
import numpy as np
import math
import scipy.stats as st
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def get_likert_probs(mean, std, scale_min, scale_max):
    """Generate discrete probabilities that match target mean and std for Likert scale."""
    scale_size = scale_max - scale_min + 1
    probs = np.ones(scale_size) / scale_size
    
    def objective(probs):
        x = np.arange(scale_min, scale_max + 1)
        current_mean = np.sum(x * probs)
        current_std = np.sqrt(np.sum((x - current_mean)**2 * probs))
        return (current_mean - mean)**2 + (current_std - std)**2
    
    constraints = ({'type': 'eq', 'fun': lambda p: np.sum(p) - 1})  # probabilities sum to 1
    bounds = [(0, 1) for _ in range(scale_size)]  # probabilities between 0 and 1
    result = minimize(objective, probs, method='SLSQP', bounds=bounds, constraints=constraints)
    return np.array(result.x)

def sample_batch(size, dist_type, **kwargs):
    """Draw samples from the specified distribution."""
    if dist_type == 'likert':
        probs = kwargs.get('probs')
        scale_min = kwargs.get('scale_min')
        scale_max = kwargs.get('scale_max')
        samples = np.random.choice(np.arange(scale_min, scale_max + 1), size=size, p=probs)
    elif dist_type == 'bernoulli':
        p = kwargs.get('p')
        samples = np.random.binomial(1, p, size)
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")
    
    if kwargs.get('verbose', False):
        print(f"Samples: {samples}")
    return samples

def half_width_likert(s, n, alpha=0.05):
    """Two‑sided t‑interval half‑width for the mean (unknown variance)."""
    tcrit = st.t.ppf(1 - alpha / 2, df=n - 1)
    return tcrit * s / math.sqrt(n)

def wilson_half_width(x, n, alpha=0.05):
    """Half‑width of the (1‑alpha) Wilson score CI for a proportion."""
    if n == 0:
        return 0.5  # arbitrary
    z = st.norm.ppf(1 - alpha / 2)
    phat = x / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2*n)) / denom
    half = z * math.sqrt((phat*(1-phat) + z**2 / (4*n)) / n) / denom
    return half

def next_chunk_size(n_current, n_forecast, min_chunk, max_chunk):
    """Decide how many extra observations to draw next."""
    gap = n_forecast - n_current
    return max(min_chunk, min(max_chunk, gap))

def expected_n(sigma, K, df, alpha):
    """
    Expected sample size for a bounded discrete scale given:
        sigma = σ            (s.d.)
        K     = number of equal classes
    Uses the d = df/(safety_factor*K) heuristic.
    """
    safety_factor = 3

    z  = st.norm.ppf(1 - alpha / 2)
    d  = df / (safety_factor * K)       # precision target

    return math.ceil((z * sigma / d) ** 2)


def run_sequential_sampling(dist_type, params, verbose=False):
    """
    Run sequential sampling until precision target is met.
    
    Parameters:
    -----------
    dist_type : str
        'likert' or 'bernoulli'
    target_params : dict
        Parameters for the target distribution
    sampling_params : dict
        Parameters for the sampling process
    verbose : bool
        Whether to print detailed progress
    
    Returns:
    --------
    dict
        Results of the sampling process
    """
    # Unpack sampling parameters
    n0 = params['sampling']['n0']
    min_chunk = params['sampling']['min_chunk']
    max_chunk = params['sampling']['max_chunk']
    alpha = 1 - params['sampling']['confidence']
    decision_classes = params[dist_type]['classes']
    
    # Prepare distribution-specific parameters
    if dist_type == 'likert':
        mean = params[dist_type]['mean']
        std = params[dist_type]['std']
        scale_min = params[dist_type]['scale_min']
        scale_max = params[dist_type]['scale_max']
        probs = get_likert_probs(mean, std, scale_min, scale_max)
        if verbose:
            # Format probabilities as a map of values to probabilities
            prob_map = {i: f"{p:.4f}" for i, p in enumerate(probs, start=scale_min)}
            print("Likert probability distribution:")
            for value, prob in prob_map.items():
                print(f"  {value}: {prob}")
        
        sampling_kwargs = {
            'probs': probs,
            'scale_min': scale_min,
            'scale_max': scale_max,
            'verbose': verbose
        }
    elif dist_type == 'bernoulli':
        p = params[dist_type]['p']
        scale_min = 0
        scale_max = 1
        sampling_kwargs = {
            'p': p,
            'scale_min': scale_min,
            'scale_max': scale_max, 
            'verbose': verbose
        }
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")
    
    # Start sampling
    d = (scale_max - scale_min) / (3 * decision_classes)

    all_samples = list(sample_batch(n0, dist_type, **sampling_kwargs))
    iteration = 1
    
    while True:
        n = len(all_samples)
        
        if dist_type == 'likert':
            xbar = np.mean(all_samples)
            s = np.std(all_samples, ddof=1)
            h = half_width_likert(s, n, alpha)
            if verbose:
                print(f"\nIteration {iteration:2d} | n = {n:3d} | mean = {xbar:.3f} | "
                      f"sd = {s:.3f} | half‑width = {h:.3f}")
            
        elif dist_type == 'bernoulli':
            x = sum(all_samples)
            phat = x / n
            s = math.sqrt(phat * (1 - phat))
            h = wilson_half_width(x, n, alpha)
            if verbose:
                print(f"\nIter {iteration:2d} | n = {n:4d} | phat = {phat:.4f} | Wilson half‑w = {h:.4f}")
            
        # Check if precision target is met
        if h <= d:
            if verbose:
                print("\nPrecision target met — stop sampling.")
            break

        n_forecast = expected_n(
            s, 
            params[dist_type]['classes'],
            params[dist_type]['range'],
            alpha
        )
        
        # Decide next chunk size and collect more samples
        chunk = next_chunk_size(n, n_forecast, min_chunk, max_chunk)
        
        if verbose:
            print(f"  forecast n_total ≈ {n_forecast}, drawing next chunk of {chunk} …")
        
        all_samples.extend(sample_batch(chunk, dist_type, **sampling_kwargs))
        iteration += 1
    
    # Prepare results
    results = {
        'iterations': iteration,
        'sample_size': len(all_samples),
        'all_samples': all_samples
    }
    
    if dist_type == 'likert':
        results.update({
            'mean': np.mean(all_samples),
            'std': np.std(all_samples, ddof=1),
            'half_width': half_width_likert(np.std(all_samples, ddof=1), len(all_samples), alpha)
        })
    elif dist_type == 'bernoulli':
        results.update({
            'phat': sum(all_samples) / len(all_samples),
            'half_width': wilson_half_width(sum(all_samples), len(all_samples), alpha)
        })
    
    return results

def run_multiple_iterations(iterations, dist_type, params, verbose=False):
    """Run multiple iterations of sequential sampling and report statistics."""
    sample_sizes = []
    num_iterations = []
    
    for i in range(iterations):
        if verbose and i % 100 == 0:
            print(f"Running iteration {i+1}/{iterations}")
        
        results = run_sequential_sampling(dist_type, params, verbose=False)
        sample_sizes.append(results['sample_size'])
        num_iterations.append(results['iterations'])
    
    stats = {
        'iterations': {
            'mean': np.mean(num_iterations),
            'std': np.std(num_iterations),
            'min': np.min(num_iterations),
            'max': np.max(num_iterations)
        },
        'sample_sizes': {
            'mean': np.mean(sample_sizes),
            'std': np.std(sample_sizes),
            'min': np.min(sample_sizes),
            'max': np.max(sample_sizes)
        }
    }
    
    return stats

def main(dist_type):
    """Main function to run the experiment with user-defined parameters."""
    print(f"Precision-based Sequential Sampling {dist_type}")
    print("--------------------------------------------")

    params = {
        'likert': {
            'mean': 8.3,
            'std': 1,
            'classes': 9,
            'range': 9,
            'scale_min': 1,
            'scale_max': 10,
        },
        'bernoulli': {
            'p': 0.2,
            'classes': 2,
            'range': 1,
            'scale_min': 0,
            'scale_max': 1,
        },
        'sampling': {
            'n0': 10,
            'confidence': 0.95,
            'min_chunk': 1,
            'max_chunk': 10,
        }
    }
    params['bernoulli']['std'] = math.sqrt(params['bernoulli']['p'] * (1 - params['bernoulli']['p']))

    single_run = False

    params['expected_n'] = expected_n(
        params[dist_type]['std'], 
        params[dist_type]['classes'],
        params[dist_type]['range'],
        (1 - params['sampling']['confidence'])
    )
    print('True std: ', params[dist_type]['std'])
    print('True mean: ', params[dist_type]['mean'])
    print('Classes: ', params[dist_type]['classes'])
    print('Range: ', params[dist_type]['range'])
    print('Confidence: ', params['sampling']['confidence'])
    print(f"Expected n: {params['expected_n']:.2f}")

    if single_run:
        # Run a single iteration with verbose output
        print("\nRunning a single sample with verbose output:")
        single_result = run_sequential_sampling(dist_type, params, verbose=True)
    
        # Print single result summary
        print("\nSingle Run Summary")
        print("-----------------")
        if dist_type == 'likert':
            print(f"Sample size: {single_result['sample_size']}")
            print(f"Expected n: {params['expected_n']:.2f}")
            print(f"Sample mean: {single_result['mean']:.3f}")
            print(f"True mean: {params['likert']['mean']:.3f}")
            print(f"Sample std: {single_result['std']:.3f}")
            print(f"Half-width: {single_result['half_width']:.3f}")
        else:  # bernoulli
            print(f"Sample size: {single_result['sample_size']}")
            print(f"Expected n: {params['expected_n']:.2f}")
            print(f"Sample phat: {single_result['phat']:.4f}")
            print(f"True p: {params['bernoulli']['p']:.4f}")
            print(f"Half-width: {single_result['half_width']:.4f}")
    
    iterations = 1000
    print(f"\nRunning {iterations} iterations...")
    
    stats = run_multiple_iterations(iterations, dist_type, params)
    
    # Print statistics
    print("\nIteration Statistics")
    print("-------------------")
    print(f"Mean iterations: {stats['iterations']['mean']:.2f}")
    print(f"Std iterations: {stats['iterations']['std']:.2f}")
    print(f"Min iterations: {stats['iterations']['min']}")
    print(f"Max iterations: {stats['iterations']['max']}")
    
    print("\nSample Size Statistics")
    print("---------------------")
    print(f"Mean sample size: {stats['sample_sizes']['mean']:.2f}")
    print(f"Std sample size: {stats['sample_sizes']['std']:.2f}")
    print(f"Min sample size: {stats['sample_sizes']['min']}")
    print(f"Max sample size: {stats['sample_sizes']['max']}")
    print(f"Expected n: {params['expected_n']:.2f}")

    print(f"--------------------------------------------\n\n")

def plot_parameter_effects(classes=9, confidence=0.95, figsize=(18, 6)):
    """
    Plot how distribution parameters affect the expected sample size.
    
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
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot 1: Std vs Sample Size
    plt.subplot(1, 3, 1)
    
    # Degrees of freedom
    df_likert = 4  # 5-1 for 5-point Likert
    
    # Likert line
    stds_likert = np.linspace(0.1, 2.0, 100)
    sample_sizes_likert = []
    
    for std in stds_likert:
        d = (4) / (3 * classes)  # Precision target
        n = expected_n(std, classes, df_likert, alpha)
        sample_sizes_likert.append(n)
    
    # Normalize std by df-1
    normalized_stds_likert = stds_likert / df_likert
    
    plt.plot(normalized_stds_likert, sample_sizes_likert, 'b-')
    
    plt.title(f'Effect of Normalized Standard Deviation\n(Likert, classes K = {classes}, confidence = {confidence})')
    plt.xlabel('normalized standard deviation δ')
    plt.ylabel('Expected Sample Size')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Effect of Confidence Intervals on Sample Size (Likert only)
    plt.subplot(1, 3, 2)
    
    # Range of confidence levels (more granular)
    confidence_range = np.linspace(0.5, 0.99, 100)
    
    # Fixed parameters for Likert
    df_likert = 4
    fixed_classes = classes
    normalized_std_likert = 0.25  # A reasonable value for Likert scales
    
    # Calculate sample sizes
    likert_n = []
    for conf in confidence_range:
        alpha_val = 1 - conf
        n = expected_n(normalized_std_likert * df_likert, fixed_classes, df_likert, alpha_val)
        likert_n.append(n)
    
    plt.plot(confidence_range, likert_n, 'b-')
    
    plt.title(f'Effect of Confidence Level\n(Likert, classes K = {fixed_classes}, normalized std δ = {normalized_std_likert})')
    plt.xlabel('Confidence Level')
    plt.ylabel('Expected Sample Size')
    plt.grid(True)
    
    # Add reference lines for common confidence levels
    for conf in [0.8, 0.9, 0.95, 0.99]:
        alpha_val = 1 - conf
        n = expected_n(normalized_std_likert * df_likert, fixed_classes, df_likert, alpha_val)
        plt.axvline(x=conf, color='gray', linestyle='--', alpha=0.7)
        plt.text(conf+0.005, n-0.5, f"{conf:.2f}", fontsize=8)
    
    # Plot 3: Effect of number of classes K on Sample Size (Likert only)
    plt.subplot(1, 3, 3)
    
    # Fixed parameters
    df_likert = 4
    normalized_std_likert = 0.25
    fixed_confidence = confidence
    fixed_alpha = 1 - fixed_confidence
    
    # Range of classes
    class_range = np.arange(2, 21)
    
    # Calculate sample sizes
    class_n = []
    for k in class_range:
        n = expected_n(normalized_std_likert * df_likert, k, df_likert, fixed_alpha)
        class_n.append(n)
    
    plt.plot(class_range, class_n, 'b-')
    
    plt.title(f'Effect of Number of Classes K\n(Likert, confidence = {fixed_confidence}, normalized std δ = {normalized_std_likert})')
    plt.xlabel('Number of Classes K')
    plt.ylabel('Expected Sample Size')
    plt.grid(True)
    plt.xticks(np.arange(2, 21, 2))
    
    plt.tight_layout()
    plt.savefig(f"{os.path.dirname(__file__)}/parameter_effects.png")
    plt.show()
    
    print(f"Plots saved to './parameter_effects.png'")

if __name__ == "__main__":
    # Add a call to the new function
    # plot_parameter_effects(classes=9)
    
    main('likert')
    # main('bernoulli')
