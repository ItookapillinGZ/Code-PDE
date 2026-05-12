import os
import pandas as pd
import time
import traceback
from code_generation import generate_and_debug, prepare_working_folder

def select_seed_implementations(total_num_sample_solvers, num_sample_for_refine=None):
    import random
    if (num_sample_for_refine is None or
        num_sample_for_refine > total_num_sample_solvers or
        num_sample_for_refine == -1):
        num_sample_for_refine = total_num_sample_solvers
    return random.sample(range(total_num_sample_solvers), num_sample_for_refine)

def refine(cfg, round_idx: int):
    """
    精炼函数：由 main.py 的外层循环调用，每次执行一个特定的 round_idx。
    """
    # 1. 配置提取
    num_trials = cfg.method.num_debugging_trials_per_sample
    pde_name = cfg.pde.name
    working_folder = cfg.working_folder
    model_name = cfg.model.name
    num_sample_for_refine = cfg.method.num_sample_for_refine
    use_sample_solver_init = cfg.method.use_sample_solver_init

    # 2. 安全检查
    assert use_sample_solver_init, 'Sample solvers must be enabled for refinement'

    # 3. 获取种子库信息
    sample_solver_folder = os.path.join('solvers', pde_name, cfg.pde.pde_setting_name, 'seeds')
    seed_results_path = os.path.join(sample_solver_folder, 'seed_results.csv')

    if not os.path.exists(seed_results_path):
        raise FileNotFoundError(f"找不到种子结果文件: {seed_results_path}")

    sample_solver_info = pd.read_csv(seed_results_path)
    total_num_sample_solvers = len(sample_solver_info)

    # 4. 目录初始化（仅在整个流程的第一轮执行）
    # 注意：这里判断 main 传入的 round_idx 是否为 0
    if round_idx == 0:
        print(f"\n[Init] 初始化工作目录: {working_folder}")
        prepare_working_folder(
            cfg,
            working_folder=working_folder,
            pde_name=pde_name,
            use_sample_solver_init=use_sample_solver_init
        )

    # 5. 执行当前轮次的 Refine 逻辑
    # 删除了内部的 for 循环，由外层 main.py 控制进度
    print(f"\n" + "=" * 30)
    print(f"正在执行 Refine 任务 - Round {round_idx}")
    print("=" * 30)

    try:
        # 选择参考种子
        seed_indices = select_seed_implementations(
            total_num_sample_solvers=total_num_sample_solvers,
            num_sample_for_refine=num_sample_for_refine
        )
        print(f">>> 选中参考种子索引: {seed_indices}")

        # 调用核心生成与调试函数
        # 传入 sample_idx=0 解决之前的 TypeError
        generate_and_debug(
            cfg,
            round_idx=round_idx,
            sample_idx=0,
            num_trials=num_trials,
            pde_name=pde_name,
            working_folder=working_folder,
            seed_implementations=seed_indices,
            model_name=model_name
        )

        print(f">>> Round {round_idx} 迭代成功结束。")

    except Exception as e:
        print(f"!!! Round {round_idx} 运行发生严重错误!")
        traceback.print_exc()

    # 稍微延迟，保护 API 频率
    time.sleep(cfg.get('api_delay', 3))