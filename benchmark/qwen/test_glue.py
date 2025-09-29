from datasets import load_dataset


ds = load_dataset("nyu-mll/glue", "qnli")
ds2 = load_dataset("tatsu-lab/alpaca")

print(ds)
print(ds2)