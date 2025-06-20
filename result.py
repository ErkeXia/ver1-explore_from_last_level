import json

with open('results.json') as f:
    data = json.load(f)

total = len(data)
true_count = sum(1 for item in data if item.get('correct') is True)

true_rate = true_count / total if total > 0 else 0
print(f"True rate: {true_rate:.2%}")
