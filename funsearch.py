import math
import os
import pandas as pd
import shutil
import time
import logging

from code_generation import generate_and_debug, prepare_working_folder, code_execution, get_results
from program_database import ProgramsDatabase, ProgramsDatabaseConfig

# 设置日志，方便排查为什么 0_1 等文件消失
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_seed_score(nRMSE, convergence_rate):
    """计算 FunSearch 的多维评分，用于程序存入数据库时的排序"""
    # 处理失败情况
    if nRMSE is None or math.isnan(nRMSE):
        return {'bucketed_convergence_rate': 0, 'bucketed_nRMSE': 0}

    return {
        'bucketed_convergence_rate': int(max(0, convergence_rate) * 4),
        'bucketed_nRMSE': int(-math.log10(min(1e9, max(1e-12, nRMSE))) * 10)
    }


def funsearch(cfg, round_idx=None):
    # 配置参数获取
    num_trials = cfg.method.num_debugging_trials_per_sample
    pde_name = cfg.pde.name
    working_folder = cfg.working_folder
    model_name = cfg.model.name
    num_search_rounds = cfg.method.num_search_rounds
    num_samples_per_round = getattr(cfg.method, 'num_samples_per_round', 2)  # 解决你 0_1 消失的问题
    num_initial_seeds = cfg.method.num_initial_seeds
    use_sample_solver_init = cfg.method.use_sample_solver_init

    prepare_working_folder(
        cfg,
        working_folder=working_folder,
        pde_name=pde_name,
        use_sample_solver_init=use_sample_solver_init
    )

    # 初始化程序数据库
    pd_cfg = ProgramsDatabaseConfig()
    program_db = ProgramsDatabase(pd_cfg)

    # --- 第一阶段：加载初始种子 (Seeds) ---
    # 这里逻辑保持原样，将之前运行好的优秀代码注册进数据库作为“祖先”
    # [此处省略你原有的 seed 加载逻辑，确保 register_program 被正确调用]

    # --- 第二阶段：FunSearch 演化循环 ---
    logging.info(f"开始 FunSearch 演化，共 {num_search_rounds} 轮")

    for round_idx in range(num_initial_seeds, num_initial_seeds + num_search_rounds):
        for sample_idx in range(num_samples_per_round):
            # 核心：从数据库获取当前最优的一组种子作为 Prompt Context
            island_id, seed_ids = program_db.get_seed()

            # 组合标识符，例如 1_0, 1_1
            current_tag = f"{round_idx}_{sample_idx}"
            logging.info(f"正在执行 Round {round_idx}, Sample {sample_idx}...")

            try:
                # 调用模型生成新代码：这里会将 seed_ids 里的代码加入 Prompt
                relative_error, elapsed_time, avg_rate = generate_and_debug(
                    cfg,
                    round_idx=current_tag,  # 传入带 sample 标识的索引
                    num_trials=num_trials,
                    pde_name=pde_name,
                    working_folder=working_folder,
                    seed_implementations=seed_ids,  # FunSearch 的关键
                    model_name=model_name
                )

                # 评估新生成的程序
                seed_score = get_seed_score(relative_error, avg_rate)

                # 读取生成的代码长度
                impl_path = os.path.join(working_folder, f'implementation_{current_tag}.py')
                if os.path.exists(impl_path):
                    with open(impl_path, 'r', encoding='utf-8') as f:
                        program_len = len(f.readlines())

                    # 将表现优秀的新程序“进化”回数据库
                    program_db.register_program(
                        program=current_tag,
                        program_len=program_len,
                        island_id=island_id,
                        scores_per_test=seed_score,
                    )
                    logging.info(f"Sample {current_tag} 注册成功，nRMSE: {relative_error}")

            except Exception as e:
                logging.error(f"Round {current_tag} 出错: {e}")
                continue

    # --- 第三阶段：汇报最终结果 ---
    report_final_best(cfg, working_folder, pde_name)


def report_final_best(cfg, working_folder, pde_name):
    """筛选所有实验中表现最好的代码进行最终验证"""
    results_path = os.path.join(working_folder, 'test_results.csv')
    if not os.path.exists(results_path):
        return

    results = pd.read_csv(results_path)
    # 按照 nRMSE 最小、耗时最短排序
    keywords = ['nRMSE', 'elapsed_time', 'convergence_rate']
    for k in keywords:
        results[k] = pd.to_numeric(results[k], errors="coerce")

    sorted_results = results.sort_values(by=['nRMSE', 'elapsed_time'], ascending=[True, True])
    best_row = sorted_results.iloc[0]
    best_tag = best_row["round"]  # 这里的 round 现在可能是 "1_0" 字符串

    logging.info(f"最终胜出者标识: {best_tag}")

    # 复制并执行最终测试（测试集评估）
    test_run_id = "final_eval"
    shutil.copy(
        os.path.join(working_folder, f'implementation_{best_tag}.py'),
        os.path.join(working_folder, f'implementation_{test_run_id}.py')
    )

    # ... 执行最终 evaluation 逻辑 (参考你原有的 code_execution) ...