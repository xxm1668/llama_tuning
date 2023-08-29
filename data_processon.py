import json

filename = r'/Users/haojingkun/PycharmProjects/llama_tuning/data/bankuai.json'
target_filename = r'/Users/haojingkun/PycharmProjects/llama_tuning/data/bankuai2.json'
target_w = open(target_filename, 'a+', encoding='utf-8')
with open(filename, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        print(line)
        data_json = json.loads(line)
        question = data_json['instruction']
        answer = data_json['output']
        print(len(answer))
        tmp = {}
        tmp['instruction'] = question.replace(' ', '')
        tmp['input'] = ''
        tmp['output'] = answer.replace(' ', '')
        tmp['history'] = []
        target_w.write(json.dumps(tmp, ensure_ascii=False) + '\n')
