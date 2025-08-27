from datasets import load_dataset


dataset = load_dataset("squad")

print(dataset["train"][0])
