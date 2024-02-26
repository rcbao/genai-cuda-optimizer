import argparse
from openai import OpenAI
import os

# Initialize parser
parser = argparse.ArgumentParser(description="Optimize CUDA code using OpenAI API.")

# Adding arguments
parser.add_argument("cuda_code_file", type=str, help="Path to the CUDA code file")
parser.add_argument("llm_prompt_file", type=str, help="Path to the LLM prompt file")
parser.add_argument("model_name", type=str, help="OpenAI model name")
parser.add_argument("cuda_version", type=str, help="CUDA version for compatibility")
parser.add_argument(
    "optimization_level",
    type=int,
    choices=range(0, 6),
    help="Optimization level (0 to 5)",
)
parser.add_argument("output_folder", type=str, help="Output folder path")

# Parse arguments
args = parser.parse_args()

# OpenAI API key setup (replace YOUR_API_KEY with your actual OpenAI API key)
openai_api_key = "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"
client = OpenAI(api_key=openai_api_key)

optimization_text = {
    1: "Optimize for readability while ensuring minimal performance trade-offs",
    2: "Optimize for performance with minimal impact on readability",
}


# Function to read file content
def read_file(file_path):
    with open(file_path, "r") as file:
        return file.read()


# Read the CUDA code and LLM prompt file
cuda_code = read_file(args.cuda_code_file)
llm_prompt = read_file(args.llm_prompt_file)

# Prepare the prompt for the API
full_prompt = f"{llm_prompt}\n\nCUDA Version: {args.cuda_version}\nOptimization Level: {args.optimization_level}\n\n{cuda_code}"


# Call OpenAI API to optimize the CUDA code
try:
    response = client.chat.completions.create(
        model=args.model_name, prompt=full_prompt, temperature=0, max_tokens=2048
    )

    optimized_code = response.choices[0].text.strip()

    # Determine the new file name
    original_file_name = os.path.basename(args.cuda_code_file)
    new_file_name = f"optimized-{os.path.splitext(original_file_name)[0]}-o{args.optimization_level}.cu"
    output_path = os.path.join(args.output_folder, new_file_name)

    # Write the optimized code to the new file
    with open(output_path, "w") as file:
        file.write(optimized_code)

    print(f"Optimized CUDA code has been saved to {output_path}")

except Exception as e:
    print(f"An error occurred: {e}")
