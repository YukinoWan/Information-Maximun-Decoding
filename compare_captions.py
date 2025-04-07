import json

iter = [1, 2, 3, 4, 5]

datas = []
for i in iter:
    with open(f"evaluation/test_qwen2-caption-{i}/res_mmau_mini.json", "r") as f:
        data =json.load(f)
        datas.append(data)

print(len(datas[0]))
for i in range(3):
    for data in datas:
        print(data[i]["model_prediction"])
        print("-" * 100)
    print("=" * 100)

