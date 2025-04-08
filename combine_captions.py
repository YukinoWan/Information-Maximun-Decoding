import json

iter = [1, 2, 3, 4, 5]

datas = []
for i in iter:
    with open(f"caption/test_qwen2-caption-with-question-{i}/res_mmau_mini.json", "r") as f:
        data =json.load(f)
        datas.append(data)

print(len(datas[0]))
output_datas = []
for i in range(len(datas[0])):
    tmp_dict = datas[0][i]
    captions = []
    for data in datas:
        captions.append(data[i]["model_prediction"])
        # print(data[i]["model_prediction"])
        # print("-" * 100)
    tmp_dict["model_prediction"] = captions
    output_datas.append(tmp_dict)

with open("caption/test_qwen2-5-caption-with-question/res_mmau_mini.json", "w") as f:
    json.dump(output_datas, f, indent=4)

