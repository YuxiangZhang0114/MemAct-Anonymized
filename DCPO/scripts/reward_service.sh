python3 -m sglang.launch_server  \
    --model /models/gpt-oss-120b   \
    --served-model-name gpt-oss  \
    --mem-fraction-static 0.8   \
    --max-running-requests 64   \
    --host 0.0.0.0   \
    --port 9527   \
    --tool-call-parse 'gpt-oss'   \
    --context-length 8192