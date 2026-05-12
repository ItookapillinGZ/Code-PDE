# from anthropic import Anthropic
# from google import genai
# from google.genai import types
# from openai import OpenAI
# import time
#
#
# def get_client(messages, cfg):
#     if 'gpt' in cfg.model.family_name or cfg.model.family_name == 'o':
#         client = OpenAI(api_key=cfg.model.api_key)
#     elif 'claude' in cfg.model.family_name:
#         client = Anthropic(base_url=cfg.model.ANTHROPIC_BASE_URL)
#     elif 'deepseek' in cfg.model.family_name:
#         client = OpenAI(api_key=cfg.model.api_key, base_url=cfg.model.base_url)
#     elif 'gemini' in cfg.model.family_name:
#         client = genai.Client(api_key=cfg.model.api_key)
#     elif cfg.model.family_name == 'qwen':
#         client = OpenAI(api_key=cfg.model.api_key, base_url=cfg.model.base_url)
#     else:
#         raise ValueError(f'Model {cfg.model.family_name} not recognized')
#     return client
#
#
# def generate_response(messages, cfg):
#     client = get_client(messages, cfg)
#     model_name = cfg.model.name
#     if 'o1' in model_name or 'o3' in model_name or 'o4' in model_name:
#         # Need to follow the restrictions
#         if 'o1' in model_name and len(messages) > 0 and messages[0]['role'] == 'system':
#             system_prompt = messages[0]['content']
#             messages = messages[1:]
#             messages[0]['content'] = system_prompt + messages[0]['content']
#         # TODO: add these to the hydra config
#         num_tokens = 16384
#         temperature = 1.0
#         start_time = time.time()
#         response = client.chat.completions.create(
#             model=model_name,
#             messages=messages,
#             max_completion_tokens=num_tokens,
#             temperature=temperature)
#         end_time = time.time()
#         print(f'It takes {model_name} {end_time - start_time:.2f}s to generate the response.')
#         return response
#
#     if 'claude' in model_name:
#         num_tokens = 8192  # Give claude more tokens
#         temperature = 0.7
#
#         if len(messages) > 0 and messages[0]['role'] == 'system':
#             system_prompt = messages[0]['content']
#             messages = messages[1:]
#
#         start_time = time.time()
#         if cfg.model.thinking:
#             num_thinking_tokens = 12288
#             response = client.messages.create(
#                 model=model_name,
#                 max_tokens=num_tokens + num_thinking_tokens,
#                 thinking={"type": "enabled", "budget_tokens": num_thinking_tokens},
#                 system=system_prompt,
#                 messages=messages,
#                 # temperature has to be set to 1 for thinking
#                 temperature=1.0,
#             )
#         else:
#             response = client.messages.create(
#                 model=model_name,
#                 max_tokens=num_tokens,
#                 system=system_prompt,
#                 messages=messages,
#                 temperature=temperature,
#             )
#         end_time = time.time()
#         print(f'It takes {model_name} {end_time - start_time:.2f}s to generate the response.')
#         return response
#
#     if 'gemini' in model_name:
#         start_time = time.time()
#         if len(messages) > 0 and messages[0]['role'] == 'system':
#             # If the first message is a system message, we need to prepend it to the user message
#             system_prompt = messages[0]['content']
#             messages = messages[1:]
#             messages[0]['content'] = system_prompt + messages[0]['content']
#
#         for message in messages:
#             if message['role'] == 'assistant':
#                 message['role'] = 'model'
#
#         chat = client.chats.create(
#             model=model_name,
#             history=[
#                 types.Content(role=message['role'], parts=[types.Part(text=message['content'])])
#                 for message in messages[:-1]
#             ],
#         )
#         response = chat.send_message(message=messages[-1]['content'])
#         end_time = time.time()
#         print(f'It takes {model_name} {end_time - start_time:.2f}s to generate the response.')
#         return response
#
#     num_tokens = 4096
#     temperature = 0.7
#
#     start_time = time.time()
#     response = client.chat.completions.create(
#         model=model_name,
#         messages=messages,
#         max_tokens=num_tokens,
#         temperature=temperature,
#         stream=('qwq' in model_name),
#     )
#
#     if 'qwq' in model_name:
#         answer_content = ""
#         for chunk in response:
#             if chunk.choices:
#                 delta = chunk.choices[0].delta
#                 if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
#                     # We don't need to print the reasoning content
#                     pass
#                 else:
#                     answer_content += delta.content
#         response = answer_content
#
#     end_time = time.time()
#     print(f'It takes {model_name} {end_time - start_time:.2f}s to generate the response.')
#     return response
import os
import time
from anthropic import Anthropic
from google import genai
from google.genai import types
from openai import OpenAI
from dotenv import load_dotenv

# --- 关键修改：加载 .env 环境 ---
# 这会让代码自动从根目录寻找并加载 ANTHROPIC_API_KEY 等变量
load_dotenv(override=True)


def get_client(messages, cfg):
    # 逻辑：优先从 Hydra 配置(cfg)读，如果没有，则读取环境变量
    api_key = getattr(cfg.model, 'api_key', os.getenv("ANTHROPIC_API_KEY"))

    if 'gpt' in cfg.model.family_name or cfg.model.family_name == 'o':
        client = OpenAI(api_key=api_key)

    elif 'claude' in cfg.model.family_name:
        # 兼容 s01 的逻辑：优先使用配置里的 Base URL，没有则读环境环境
        base_url = getattr(cfg.model, 'ANTHROPIC_BASE_URL', os.getenv("ANTHROPIC_BASE_URL"))
        # 如果提供了 base_url，则初始化时带上；否则直接初始化（SDK 会自动读环境变量里的 API_KEY）
        client = Anthropic(api_key=api_key, base_url=base_url)

    elif 'deepseek' in cfg.model.family_name:
        ds_base = getattr(cfg.model, 'base_url', "https://api.deepseek.com")
        client = OpenAI(api_key=api_key, base_url=ds_base)

    elif 'gemini' in cfg.model.family_name:
        gemini_key = getattr(cfg.model, 'api_key', os.getenv("GEMINI_API_KEY"))
        client = genai.Client(api_key=gemini_key)

    elif cfg.model.family_name == 'qwen':
        qwen_base = getattr(cfg.model, 'base_url', os.getenv("DASHSCOPE_BASE_URL"))
        client = OpenAI(api_key=api_key, base_url=qwen_base)

    else:
        raise ValueError(f'Model {cfg.model.family_name} not recognized')
    return client


def generate_response(messages, cfg):
    client = get_client(messages, cfg)
    model_name = cfg.model.name

    # 1. 处理 OpenAI o1/o3/o4 系列
    if 'o1' in model_name or 'o3' in model_name or 'o4' in model_name:
        if 'o1' in model_name and len(messages) > 0 and messages[0]['role'] == 'system':
            system_prompt = messages[0]['content']
            messages = messages[1:]
            messages[0]['content'] = system_prompt + "\n" + messages[0]['content']

        num_tokens = 16384
        temperature = 1.0
        start_time = time.time()
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=num_tokens,
            temperature=temperature
        )
        print(f'It takes {model_name} {time.time() - start_time:.2f}s to generate the response.')
        return response

    # 2. 处理 Claude 系列
    if 'claude' in model_name:
        num_tokens = 8192
        temperature = 0.7
        system_prompt = ""

        # Claude 的 SDK 将 system prompt 与 messages 分开传入
        if len(messages) > 0 and messages[0]['role'] == 'system':
            system_prompt = messages[0]['content']
            messages = messages[1:]

        start_time = time.time()
        if getattr(cfg.model, 'thinking', False):
            num_thinking_tokens = 12288
            response = client.messages.create(
                model=model_name,
                max_tokens=num_tokens + num_thinking_tokens,
                thinking={"type": "enabled", "budget_tokens": num_thinking_tokens},
                system=system_prompt,
                messages=messages,
                temperature=1.0,
            )
        else:
            response = client.messages.create(
                model=model_name,
                max_tokens=num_tokens,
                system=system_prompt,
                messages=messages,
                temperature=temperature,
            )
        print(f'It takes {model_name} {time.time() - start_time:.2f}s to generate the response.')
        return response

    # 3. 处理 Gemini 系列
    if 'gemini' in model_name:
        start_time = time.time()
        if len(messages) > 0 and messages[0]['role'] == 'system':
            system_prompt = messages[0]['content']
            messages = messages[1:]
            messages[0]['content'] = system_prompt + "\n" + messages[0]['content']

        # Gemini 角色映射
        for message in messages:
            if message['role'] == 'assistant':
                message['role'] = 'model'

        chat = client.chats.create(
            model=model_name,
            history=[
                types.Content(role=message['role'], parts=[types.Part(text=message['content'])])
                for message in messages[:-1]
            ],
        )
        response = chat.send_message(message=messages[-1]['content'])
        print(f'It takes {model_name} {time.time() - start_time:.2f}s to generate the response.')
        return response

    # 4. 默认处理 (DeepSeek, Qwen 等兼容 OpenAI 格式的模型)
    num_tokens = 4096
    temperature = 0.7
    start_time = time.time()

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=num_tokens,
        temperature=temperature,
        stream=('qwq' in model_name),
    )

    if 'qwq' in model_name:
        answer_content = ""
        for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if not (hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None):
                    if hasattr(delta, 'content') and delta.content:
                        answer_content += delta.content
        response = answer_content

    print(f'It takes {model_name} {time.time() - start_time:.2f}s to generate the response.')
    return response