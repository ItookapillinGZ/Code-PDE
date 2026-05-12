import os
import sys
import hydra
import re
from omegaconf import OmegaConf

# 确保这些模块与 main.py 在同一目录或已正确安装
from refine import refine
from repeated_sample import repeated_sample
from funsearch import funsearch


def get_nrmse(working_folder, round_idx, sample_idx):
    """
    从磁盘日志文件中提取 nRMSE 数值
    """
    # 匹配文件名如：output_0_0.txt
    file_path = os.path.join(working_folder, f'output_{round_idx}_{sample_idx}.txt')
    if not os.path.exists(file_path):
        return float('inf')

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # 使用正则提取日志中记录的 nRMSE 结果
            match = re.search(r'nRMSE:\s*([0-9.eE+-]+)', content)
            if match:
                return float(match.group(1))
    except Exception as e:
        print(f"Error parsing log {file_path}: {e}")

    return float('inf')


def prepare_feedback(working_folder, round_idx, best_sample_idx):
    """
    将本轮表现最好的代码和结果日志组装成给下一轮的反馈字符串
    """
    code_path = os.path.join(working_folder, f'implementation_{round_idx}_{best_sample_idx}.py')
    log_path = os.path.join(working_folder, f'output_{round_idx}_{best_sample_idx}.txt')

    feedback = f"\n### FEEDBACK FROM ROUND {round_idx} ###\n"

    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            feedback += f"Previous Execution Log:\n{f.read()}\n"

    if os.path.exists(code_path):
        with open(code_path, 'r', encoding='utf-8') as f:
            feedback += f"Previous Best Code:\n```python\n{f.read()}\n```\n"

    # 假设你在 Python 中已经计算好了当前的 nrmse 和 avg_rate
    # 以及从日志中提取的 u_min 值

        # 在引号前添加 r，将字符串转为 raw string，避免 LaTeX 反斜杠引起 SyntaxWarning
    feedback += r"""### Numerical Performance Analysis & Refinement Instruction
        Your previous solution resulted in a specific performance profile. Please analyze your logs (nRMSE and avg convergence rate) and apply the following refinement logic:

        #### [Condition 1: If Average Convergence Rate < 0 or u_min < 0]
        **Diagnosis:** Numerical Instability & Physical Violation. The error increases as dt decreases, indicating the solution is escaping its physical domain ([0, 1]).
        **Mandatory Actions:**
        1. **Value Clipping**: You MUST apply `u = np.clip(u, 0, 1)` immediately before the reaction solver to suppress Gibbs oscillations.
        2. **Symmetric Strang Splitting**: Ensure the sequence is exactly $D(\Delta t/2) \to R(\Delta t) \to D(\Delta t/2)$.
        3. **Safe Denominator**: Add a robust epsilon (e.g., 1e-15) to the analytical formula to prevent overflow.

        #### [Condition 2: If 0 < Average Convergence Rate < 1.8 or nRMSE > 1e-2]
        **Diagnosis:** Sub-optimal Convergence & Precision. You have stability but are missing the theoretical $O(\Delta t^2)$ accuracy, or have a constant bias in nRMSE.
        **Mandatory Actions:**
        1. **Spectral De-aliasing (2/3 Rule)**: Verify that high-frequency Fourier coefficients are zeroed out to prevent spectral pollution.
        2. **Full Double Precision**: Cast all arrays, FFT operations, and wavenumbers strictly to `float64` or `complex128`.
        3. **Wavenumber Scaling**: Re-check your $k$ vector definition. Ensure it aligns with the domain length $L$ and the spectral basis used in the ground truth.
        4. **Time-Step Alignment**: Ensure the final timestamps in your output tensor perfectly match the provided `t_coordinate`.

        #### [Condition 3: If Performance is Satisfactory (Rate ≈ 2.0 and nRMSE < 1e-2)]
        **Diagnosis:** High-Fidelity Performance.
        **Mandatory Actions:**
        1. **Vectorization**: Optimize the solver to process the entire batch simultaneously.
        2. **Pre-computation**: Move static coefficients (decay factors, masks) outside the time loop.

        Please re-examine your code, especially the splitting logic and boundary condition maintenance during FFT/IFFT cycles, then provide the corrected implementation."""

    return feedback


@hydra.main(config_path='configs', config_name='default', version_base=None)
def main(cfg):
    # --- 打印基础信息 ---
    print(f'Method: {cfg.method.name}')
    print(f'Model name: {cfg.model.name}')
    print(f'PDE name: {cfg.pde.name}')
    print(f'Working folder: {cfg.working_folder}')

    if not os.path.exists(cfg.working_folder):
        os.makedirs(cfg.working_folder)

    # --- 解决 Windows 编码与重定向问题 ---
    if cfg.redirect_stdout:
        sys.stdout = open(os.path.join(cfg.working_folder, 'stdout.txt'), 'w', encoding='utf-8')
        sys.stderr = open(os.path.join(cfg.working_folder, 'stderr.txt'), 'w', encoding='utf-8')

    # --- 核心改进：解锁 Hydra 结构并启动闭环迭代 ---
    # 解除 ConfigAttributeError 限制，允许动态添加 feedback 字段
    OmegaConf.set_struct(cfg, False)

    cfg.feedback = ""  # 初始化反馈为空
    max_rounds = 3  # 执行 Round 0 和 Round 1

    for round_idx in range(max_rounds):
        print(f'\n' + '=' * 30)
        print(f'STARTING ITERATION ROUND: {round_idx}')
        print('=' * 30)

        # 执行具体方法
        if cfg.method.name[:6] == 'refine':
            refine(cfg, round_idx=round_idx)
        elif cfg.method.name == 'repeated_sample':
            # 确保 repeated_sample 函数签名已修改为接收 round_idx
            repeated_sample(cfg, round_idx=round_idx)
        elif cfg.method.name == 'funsearch':
            funsearch(cfg, round_idx=round_idx)
        else:
            raise NotImplementedError(f'Unknown method: {cfg.method.name}')

        # --- 评估与反馈环节 ---
        num_samples = getattr(cfg.method, 'num_repeated_samples', 1)

        # 搜集本轮所有 sample 的误差
        round_errors = []
        for s_idx in range(num_samples):
            err = get_nrmse(cfg.working_folder, round_idx, s_idx)
            round_errors.append(err)

        # 选出本轮最优解
        best_sample_idx = round_errors.index(min(round_errors))
        current_best_error = round_errors[best_sample_idx]

        print(f"\n[Round {round_idx} Summary]")
        print(f"Best nRMSE achieved: {current_best_error:.6e} (by Sample {best_sample_idx})")

        # 判定是否提前退出
        if current_best_error < 1e-2:
            print(">>> Success! Target accuracy reached. Terminating early.")
            break

        # 准备下一轮的“进化”素材
        if round_idx < max_rounds - 1:
            cfg.feedback = prepare_feedback(cfg.working_folder, round_idx, best_sample_idx)
            print(f">>> Feedback injected for Round {round_idx + 1}")


if __name__ == "__main__":
    # 强制 Windows 终端输出 UTF-8
    if sys.platform == "win32":
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    main()