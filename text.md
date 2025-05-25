$env:HF_ENDPOINT="https://hf-mirror.com"; python autograder.py -q q2

$env:HF_ENDPOINT="https://hf-mirror.com"; python classification.py --task topic



调整一下merge规则：只删除纯数字+纯数字，或者“连续任意位数非数字+纯数字”+“纯数字”的规则。vocab规则调整为：只删除连续大于1位的纯数字或者连续任意位数非数字+连续大于1位的纯数字


