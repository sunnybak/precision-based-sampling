from litellm import acompletion
import asyncio
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os

run_timestamp = time.strftime("%Y%m%d_%H%M%S")
os.makedirs(f"{os.path.dirname(__file__)}/{run_timestamp}", exist_ok=True)

tools = [{
    "type": "function",
    "function": {
        "name": "submit_rating",
        "description": "Submit a rating for the text",
        "parameters": {
            "type": "object",
            "properties": {
                "rating": {
                    "type": "integer",
                    "description": "The rating for the text, must be between 1 and 5"
                }
            },
            "required": [
                "rating"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
}]

with open(f"{os.path.dirname(__file__)}/text", "r") as file:
    base_text = file.read()

async def sample_rating(text):
    response = await acompletion(
        model="gpt-4.1",
        messages=[
            {"role": "user", "content": text},
            {"role": "user", "content": "Please rate the following text on a scale of 1 to 5, where 1 is the worst and 5 is the best. Submit your rating using the submit_rating function."},
        ],
        tools=tools,
    )
    token_count = response.usage.prompt_tokens
    latency = response._hidden_params['_response_ms']
    cost = response._hidden_params['response_cost']
    try:
        output = json.loads(response.choices[0].message.tool_calls[0].function.arguments)["rating"]
    except:
        output = -1

    if output == -1:
        return None

    return output, latency, token_count, cost

def plot_latency_analysis():
    # Read the results
    with open(f"{os.path.dirname(__file__)}/{run_timestamp}/latency_results.json", "r") as f:
        results = json.load(f)
    
    # Prepare data for plotting
    n_values = sorted(list(set([int(k.split('_')[1]) for k in results.keys()])))
    token_values = sorted(list(set([int(k.split('_')[3]) for k in results.keys()])))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Latency vs n (averaged across token counts)
    n_data = defaultdict(list)
    for n in n_values:
        for token in token_values:
            key = f"n_{n}_tokens_{token}"
            if key in results:
                n_data[n].extend(results[key]["latencies"])
    
    n_means = [np.mean(n_data[n]) for n in n_values]
    n_p25 = [np.percentile(n_data[n], 25) for n in n_values]
    n_p75 = [np.percentile(n_data[n], 75) for n in n_values]
    
    # Calculate error bars as absolute distances from mean
    n_yerr_lower = np.array(n_means) - np.array(n_p25)
    n_yerr_upper = np.array(n_p75) - np.array(n_means)
    
    ax1.errorbar(n_values, n_means, yerr=[n_yerr_lower, n_yerr_upper],
                fmt='o-', capsize=5, label='Mean with 25-75th percentiles')
    ax1.set_xlabel('Number of concurrent requests (n)')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('Latency vs Number of Concurrent Requests')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Latency vs tokens (averaged across n values)
    token_data = defaultdict(list)
    for token in token_values:
        for n in n_values:
            key = f"n_{n}_tokens_{token}"
            if key in results:
                token_data[token].extend(results[key]["latencies"])
    
    token_means = [np.mean(token_data[t]) for t in token_values]
    token_p25 = [np.percentile(token_data[t], 25) for t in token_values]
    token_p75 = [np.percentile(token_data[t], 75) for t in token_values]
    
    # Calculate error bars as absolute distances from mean
    token_yerr_lower = np.array(token_means) - np.array(token_p25)
    token_yerr_upper = np.array(token_p75) - np.array(token_means)
    
    ax2.errorbar(token_values, token_means, yerr=[token_yerr_lower, token_yerr_upper],
                fmt='o-', capsize=5, label='Mean with 25-75th percentiles')
    ax2.set_xlabel('Number of tokens')
    ax2.set_ylabel('Latency (ms)')
    ax2.set_title('Latency vs Number of Tokens')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{os.path.dirname(__file__)}/{run_timestamp}/latency_analysis.png")
    plt.close()


async def main():
    n_values = [1, 3, 10, 30, 100]
    token_multipliers = [1, 3, 5, 10]  # Will multiply base text by these values
    
    results = defaultdict(list)
    
    for n in n_values:
        for multiplier in token_multipliers:
            print(f"\nRunning with n={n}, token_multiplier={multiplier}")
            # Multiply the text to achieve desired token count
            text = base_text * multiplier
            
            tasks = [sample_rating(text) for _ in range(n)]
            round_results = await asyncio.gather(*tasks)
            round_results = [r for r in round_results if r is not None]
            print(f"There were {len(tasks)} tasks and {len(round_results)} valid results")
            
            if len(round_results) == 0:
                print(f"No valid results for n={n}, token_multiplier={multiplier}")
                continue
            
            latencies = [r[1] for r in round_results]
            token_counts = [r[2] for r in round_results]
            
            # Store results
            results[f"n_{n}_tokens_{multiplier*1000}"] = {
                "latencies": latencies,
                "token_counts": token_counts,
                "avg_latency": np.mean(latencies),
                "std_latency": np.std(latencies)
            }
            
            print(f"Average latency: {np.mean(latencies):.2f} ms")
            print(f"Std latency: {np.std(latencies):.2f} ms")
    
    # Save all results to a JSON file
    with open(f"{os.path.dirname(__file__)}/{run_timestamp}/latency_results.json", "w") as f:
        json.dump(results, f, indent=2)

    plot_latency_analysis()


if __name__ == "__main__":
    asyncio.run(main())