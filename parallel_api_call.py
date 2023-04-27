import time
from config import OPENAI_API_KEY
import subprocess

start = time.time()

command_line_request_str = f"""/opt/anaconda3/bin/python \
    /Users/johngearig/Documents/GitHub/Encyclopedia-GPT/api_request_parallel_processor.py \
  --requests_filepath data/full_data/idx_partial_dataset.jsonl \
  --save_filepath data/full_data/1000_parallel.jsonl \
  --request_url https://api.openai.com/v1/embeddings \
  --max_requests_per_minute 1500 \
  --max_tokens_per_minute 6250000 \
  --token_encoding_name cl100k_base \
  --max_attempts 5 \
  --logging_level 20 \
  --api_key {OPENAI_API_KEY}"""

# subprocess.run(command_line_request)
subprocess.run(command_line_request_str, text=True,shell=True)

print(f'Time: {time.time() - start}')


# command_line_request = ["/opt/anaconda3/bin/python",
#                         "api_request_parallel_processor.py",
#                         "requests_filepath",
#                         "data/full_data/partial_dataset.jsonl",
#                         "save_filepath",
#                         "data/1000_parallel.jsonl",
#                         "request_url",
#                         "https://api.openai.com/v1/embeddings",
#                         "max_requests_per_minute",
#                         "1500",
#                         "max_tokens_per_minute",
#                         "6250000",
#                         "token_encoding_name",
#                         "cl100k_base",
#                         "max_attempts",
#                         "5",
#                         "logging_level"
#                         "20",
#                         "api_key",
#                         OPENAI_API_KEY]