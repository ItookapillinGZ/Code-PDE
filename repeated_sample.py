import os
import sys
import time

from code_generation import generate_and_debug, prepare_working_folder


def repeated_sample(cfg, round_idx=0):
    num_repeated_samples = cfg.method.num_repeated_samples
    num_trials = cfg.method.num_debugging_trials_per_sample
    pde_name = cfg.pde.name
    working_folder = cfg.working_folder
    model_name = cfg.model.name

    if not os.path.exists(working_folder):
        os.makedirs(working_folder)

    # --- 错误 B 修复：重定向 stdout 时指定 utf-8 编码 ---
    if cfg.redirect_stdout:
        sys.stdout = open(os.path.join(working_folder, 'stdout.txt'), 'w', encoding='utf-8')
        sys.stderr = open(os.path.join(working_folder, 'stderr.txt'), 'w', encoding='utf-8')

    print(f'Model name: {cfg.model.name}')
    print(f'Working folder: {working_folder}')
    print(f'Current Round: {round_idx}') # 打印当前轮次方便调试

    prepare_working_folder(
        cfg,
        working_folder=working_folder,
        pde_name=pde_name,
        use_sample_solver_init=False
    )

    for sample_idx in range(num_repeated_samples):
        try:
            # --- 关键修改：修正索引传递 ---
            # 1. 真正的 round_idx 传给 round_idx 参数，确保 prompt 注入反馈
            # 2. 样本序号 sample_idx 传给 sample_idx 参数，用于文件名区分
            generate_and_debug(
                cfg,
                round_idx=round_idx,    # 这一轮是迭代的第几轮
                sample_idx=sample_idx,   # 这一轮里的第几个样本
                num_trials=num_trials,
                pde_name=pde_name,
                working_folder=working_folder,
                seed_implementations=None,
                model_name=model_name
            )
        except Exception as e:
            print(f'Error in round {round_idx} sample {sample_idx}: {e}. Move on to the next sample.')

        time.sleep(2)  # 防止 API 速率限制