import openai

# 使用你的 API Key
openai.api_key = "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"

# 读取代码文件
with open("code.txt", "r",encoding='utf-8') as input_file:
    code = input_file.read()

model_engine = "text-davinci-002"
prompt = f"请注释以下代码：\n{code}"

completions = openai.Completion.create(
    engine=model_engine,
    prompt=prompt,
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=0.5,
)

annotated_code = completions.choices[0].text

# 将注释后的代码写入文件
with open("output_file.txt", "w") as output_file:
    output_file.write(annotated_code)

