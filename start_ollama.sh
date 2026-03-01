#!/bin/bash
~/Desktop/llama.cpp/build/bin/llama-server -m ./models/text2cypher-qwen/*.gguf -ngl 99 -c 4096 --host 127.0.0.1 --port 8080 &
