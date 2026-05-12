import json
import math
import os
import re
import shutil
import signal
import subprocess
import time

from llm_api import generate_response
from prompt_files import general_prompt, pde_descriptions


# --- 错误 B 修复：保持你原来的辅助函数 ---
def file_to_string(file_path):
    with open(file_path, encoding='utf-8', errors='replace') as f:
        string = ''.join(f.readlines())
    return string


def get_last_line(output_file):
    with open(output_file, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    if not lines:
        return ""
    result_line = lines[-1]
    return result_line


def generate_pde_description(cfg, pde_name):
    if pde_name == 'advection':
        desc = pde_descriptions.advection_description
        pde_description = desc.replace("{advection_beta}", str(cfg.pde.beta))
    elif pde_name == 'burgers':
        desc = pde_descriptions.burgers_description
        pde_description = desc.replace("{burgers_nu}", str(cfg.pde.nu))
    elif pde_name == 'reacdiff1d':
        desc = pde_descriptions.reacdiff_1d_description
        pde_description = desc.replace("{reacdiff1d_nu}", str(cfg.pde.nu)).replace("{reacdiff1d_rho}", str(cfg.pde.rho))
    elif pde_name == 'cns1d':
        desc = pde_descriptions.cns1d_description
        pde_description = desc.replace("{cns1d_eta}", str(cfg.pde.eta))
    elif pde_name == 'darcy':
        pde_description = pde_descriptions.darcy_description
    elif pde_name == 'ins2d':
        pde_description = pde_descriptions.ins2d_description
    else:
        raise ValueError(f'PDE {pde_name} not recognized')
    return pde_description


def generate_initial_prompt_without_seed(cfg, pde_name):
    system_prompt = general_prompt.system_prompt
    pde_description = generate_pde_description(cfg, pde_name)
    solver_template = file_to_string(f'solvers/{pde_name}/solver_template.py')
    problem = general_prompt.code_generation_without_seed_prompt.format(
        pde_description=pde_description,
        solver_template=solver_template
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem}
    ]
    return messages


def generate_initial_prompt(cfg, seed_implementations: list, working_folder: str, pde_name: str = 'burgers'):
    system_prompt = general_prompt.system_prompt
    pde_description = generate_pde_description(cfg, pde_name)
    if cfg.method.name == 'funsearch':
        seed_folder = working_folder
    else:
        seed_folder = os.path.join('solvers', pde_name, cfg.pde.pde_setting_name, 'seeds')

    examples = [
        general_prompt.code_sample.format(
            id=example_id,
            code=file_to_string(os.path.join(seed_folder, f'implementation_{seed_id}.py')),
            code_output=get_last_line(os.path.join(seed_folder, f'output_{seed_id}.txt')),
        )
        for example_id, seed_id in enumerate(seed_implementations)
    ]
    code_samples = ''.join(examples)
    problem = general_prompt.problem_prompt.format(pde_description=pde_description, code_samples=code_samples)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem}
    ]
    return messages


# --- 重点修改：Debug Prompt 增加 sample_idx 定位 ---
def generate_debugging_prompt(round_idx: int, sample_idx: int, working_folder: str,
                              debugging_reason: str = 'execution_error'):
    # 加载带二维坐标的 messages
    with open(os.path.join(working_folder, f'messages_{round_idx}_{sample_idx}.json'), 'r', encoding='utf-8') as f:
        messages = json.load(f)

    model_response = file_to_string(os.path.join(working_folder, f'responses_{round_idx}_{sample_idx}.txt'))
    messages.append({"role": "assistant", "content": model_response})

    # 读取带二维坐标的报错日志
    code_output = file_to_string(os.path.join(working_folder, f'output_{round_idx}_{sample_idx}.txt'))[-5000:]
    errors = file_to_string(os.path.join(working_folder, f'errors_{round_idx}_{sample_idx}.txt'))[-5000:]

    if debugging_reason == 'execution_error':
        feebdack = general_prompt.debugging_execution_error_prompt.format(code_output=code_output, error_message=errors)
    else:
        feebdack = general_prompt.debugging_nan_inf_prompt.format(code_output=code_output, error_message=errors)

    messages.append({"role": "user", "content": feebdack})
    return messages


def generate_prompt(cfg, round_idx: int, sample_idx: int, working_folder: str, seed_implementations: list | None = None,
                    generation_mode: str = 'initial', pde_name: str = 'burgers'):
    if generation_mode == 'debugging_execution_error':
        prompt = generate_debugging_prompt(round_idx=round_idx, sample_idx=sample_idx, working_folder=working_folder,
                                           debugging_reason='execution_error')
    elif generation_mode == 'debugging_nan_inf':
        prompt = generate_debugging_prompt(round_idx=round_idx, sample_idx=sample_idx, working_folder=working_folder,
                                           debugging_reason='nan_inf')
    elif seed_implementations is None or len(seed_implementations) == 0:
        prompt = generate_initial_prompt_without_seed(cfg, pde_name=pde_name)
    else:
        prompt = generate_initial_prompt(cfg, seed_implementations=seed_implementations, working_folder=working_folder,
                                         pde_name=pde_name)

    # 反馈注入逻辑保持不变
    if hasattr(cfg, 'feedback') and cfg.feedback:
        # 如果是 initial 模式，prompt 是 list，我们在最后一个 user 消息注入
        if isinstance(prompt, list):
            prompt[-1][
                "content"] += "\n\n" + "=" * 20 + " IMPORTANT FEEDBACK " + "=" * 20 + "\n" + cfg.feedback + "\n" + "=" * 60
        else:
            # 兼容非 list 情况
            prompt += "\n\n" + "=" * 20 + " IMPORTANT FEEDBACK " + "=" * 20 + "\n" + cfg.feedback + "\n" + "=" * 60
    return prompt


# --- 重点修改：code_generation 增加 sample_idx 并修改保存路径 ---
def code_generation(cfg, round_idx: int, sample_idx: int, working_folder: str, seed_implementations: list | None = None,
                    generation_mode: str = 'initial', pde_name: str = 'burgers', model_name='deepseek-chat'):
    messages = generate_prompt(cfg, round_idx=round_idx, sample_idx=sample_idx, working_folder=working_folder,
                               seed_implementations=seed_implementations, generation_mode=generation_mode,
                               pde_name=pde_name)

    with open(os.path.join(working_folder, f'messages_{round_idx}_{sample_idx}.json'), 'w', encoding='utf-8') as f:
        json.dump(messages, f, ensure_ascii=False, indent=4)

    responses = generate_response(messages, cfg)

    # 保持你原来的 Claude 复杂提取逻辑
    content = ""
    if 'claude' in model_name:
        try:
            if isinstance(responses.content, list):
                for block in responses.content:
                    if hasattr(block, 'text') and block.text: content = block.text
                    if hasattr(block, 'type') and block.type == 'thinking':
                        thinking_val = getattr(block, 'thinking', "")
                        with open(os.path.join(working_folder, f'thinking_{round_idx}_{sample_idx}.txt'), 'w',
                                  encoding='utf-8') as f:
                            f.write(str(thinking_val))
                        if not content: content = thinking_val
            elif isinstance(responses.content, str):
                content = responses.content
            else:
                content = str(responses.content)
        except Exception as e:
            content = str(responses)
    elif 'gemini' in model_name:
        content = responses.text
    elif 'qwq' in model_name or isinstance(responses, str):
        content = responses
    else:
        content = responses.choices[0].message.content

    if not content: raise ValueError(f"CRITICAL: Model returned empty content.")

    with open(os.path.join(working_folder, f'responses_{round_idx}_{sample_idx}.txt'), 'w', encoding='utf-8') as f:
        f.write(content)

    # 保持你原来的正则匹配逻辑
    matches = re.findall(r'```python\n?(.*?)\n?```', content, re.DOTALL)
    if not matches: matches = re.findall(r'```\n?(.*?)\n?```', content, re.DOTALL)

    if not matches:
        if "import numpy" in content and "def " in content:
            generated_code = content
        else:
            raise ValueError(f'No code block found.')
    else:
        generated_code = max(matches, key=len)

    # --- 注入排查代码：强制让模型生成的 solver 运行时自报家门 ---
    debug_marker = f'print("\\n" + "="*20 + " EXECUTION DEBUG: ROUND {round_idx} SAMPLE {sample_idx} " + "="*20 + "\\n")\n'
    generated_code = debug_marker + generated_code
    # --------------------------------------------------------
    # 关键：文件名增加 sample_idx
    with open(os.path.join(working_folder, f'implementation_{round_idx}_{sample_idx}.py'), 'w', encoding='utf-8') as f:
        f.write(generated_code)


# --- 重点修改：code_execution 增加 sample_idx 并对齐输出文件 ---
def code_execution(cfg, working_folder: str, round_idx: int = 0, sample_idx: int = 0, pde_name: str = 'burgers',
                   eval_dataset: str = None):
    # 拷贝对应的 implementation 文件
    src_file = os.path.join(working_folder, f'implementation_{round_idx}_{sample_idx}.py')
    dst_file = os.path.join(working_folder, 'solver.py')
    shutil.copy(src_file, dst_file)

    # 关键：输出日志包含 sample_idx
    job_out = open(os.path.join(working_folder, f'output_{round_idx}_{sample_idx}.txt'), 'w', encoding='utf-8')
    job_err = open(os.path.join(working_folder, f'errors_{round_idx}_{sample_idx}.txt'), 'w', encoding='utf-8')

    if eval_dataset is None:
        raw_path = os.path.join(cfg.root_dataset_folder, cfg.pde.dataset_folder_for_eval)
        eval_dataset = raw_path.replace('\\', '/')

    python_path = "python"
    # 这里 run-id 也要对齐
    cmd = (
        f'{python_path} {working_folder}/evaluator.py --save-pth {working_folder} --run-id {round_idx}_{sample_idx} --dataset-path-for-eval "{eval_dataset}" ')

    # 超参数逻辑保持不变
    hyperparam = ""
    if pde_name == 'advection':
        hyperparam = f'--beta {cfg.pde.beta} '
    elif pde_name == 'burgers':
        hyperparam = f'--nu {cfg.pde.nu} '
    elif pde_name == 'reacdiff1d':
        hyperparam = f'--nu {cfg.pde.nu} --rho {cfg.pde.rho} '
    elif pde_name == 'cns1d':
        hyperparam = f'--eta {cfg.pde.eta} '
    elif pde_name in ['darcy', 'ins2d']:
        hyperparam = f' '

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cfg.assigned_gpu)

    process = None
    try:
        process = subprocess.Popen(f'{cmd} {hyperparam}', shell=True, stdout=job_out, stderr=job_err, text=True,
                                   env=env, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
        exit_code = process.wait(timeout=cfg.pde.timeout)
        stderr = None
        status = "completed"
    except subprocess.TimeoutExpired:
        if process and os.name == 'nt':
            subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)], capture_output=True)
        exit_code = -1
        stderr = "Timeout"
        status = "timeout"
    finally:
        if job_out: job_out.close()
        if job_err: job_err.close()

    return {"exit_code": exit_code, "stderr": stderr, "status": status}


# --- 后续函数逻辑对齐 ---
def get_results(output_file):
    """
    从输出文件中读取结果。
    改进逻辑：不再仅仅读取最后一行，而是从后往前查找包含 'nRMSE:' 的行。
    """
    with open(output_file, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    # 从后往前找包含关键信息的行
    result_line = ""
    for line in reversed(lines):
        if "nRMSE:" in line:
            result_line = line
            break

    if not result_line:
        raise ValueError(f"Could not find result line in {output_file}. Check your evaluator output format.")

    try:
        # 使用更灵活的正则匹配，不再强求 \t
        nrmse_match = re.search(r'nRMSE:\s*([0-9\.eE\-\+]+)', result_line)
        time_match = re.search(r'Time:\s*([0-9\.eE\-\+]+)s', result_line)
        rate_match = re.search(r'Average convergence rate:\s*([0-9\.eE\-\+]+)', result_line)

        relative_error = float(nrmse_match.group(1)) if nrmse_match else 0.0
        elapsed_time = float(time_match.group(1)) if time_match else 0.0
        avg_rate = float(rate_match.group(1)) if rate_match else 0.0

        return relative_error, elapsed_time, avg_rate
    except Exception as e:
        print(f"Regex matching error on line: {result_line}")
        raise e


def prepare_working_folder(cfg, working_folder, pde_name='burgers', use_sample_solver_init=False):
    result_sheet_path = os.path.join(working_folder, 'test_results.csv')
    with open(result_sheet_path, 'w', encoding='utf-8') as f:
        f.write('round_sample,nRMSE,elapsed_time,convergence_rate,num_trial\n')
    shutil.copy(os.path.join('solvers', pde_name, 'evaluator.py'), os.path.join(working_folder, 'evaluator.py'))


def generate_and_debug(cfg, round_idx: int, sample_idx: int, num_trials: int, pde_name: str, working_folder: str,
                       seed_implementations: list | None, model_name: str):
    generation_mode = 'initial'
    for num_trial in range(1, num_trials + 1):
        code_generation(cfg, round_idx=round_idx, sample_idx=sample_idx, working_folder=working_folder,
                        seed_implementations=seed_implementations, generation_mode=generation_mode, pde_name=pde_name,
                        model_name=model_name)

        execution_results = code_execution(cfg, working_folder=working_folder, round_idx=round_idx,
                                           sample_idx=sample_idx, pde_name=pde_name)

        if execution_results['exit_code'] != 0:
            if num_trial < num_trials:
                generation_mode = 'debugging_execution_error'
            else:
                with open(os.path.join(working_folder, 'test_results.csv'), 'a', encoding='utf-8') as f:
                    f.write(f'{round_idx}_{sample_idx},failed,failed,failed,{num_trial}\n')
                raise ValueError(f'Error in round {round_idx}, sample {sample_idx} execution.')
        else:
            relative_error, elapsed_time, avg_rate = get_results(
                os.path.join(working_folder, f'output_{round_idx}_{sample_idx}.txt'))
            if (math.isnan(relative_error) or math.isinf(relative_error)) and num_trial < num_trials:
                generation_mode = 'debugging_nan_inf'
            else:
                with open(os.path.join(working_folder, 'test_results.csv'), 'a', encoding='utf-8') as f:
                    f.write(f'{round_idx}_{sample_idx},{relative_error},{elapsed_time},{avg_rate},{num_trial}\n')
                return relative_error, elapsed_time, avg_rate
    return None, None, None