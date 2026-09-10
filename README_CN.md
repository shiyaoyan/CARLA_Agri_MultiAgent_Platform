# CARLA 农业多智能体仿真平台

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English README](README.md)

本项目是一个由 CrewAI 管理、基于 CARLA 的农机自主导航仿真平台，面向
农业道路场景生成、拖拉机配置、真实轨迹跟踪、控制器参数优化和全流程农业
导航验证。

## 1. 架构说明

项目现在只有一个顶层流程：
`AgriculturalSimulationFlow`。该流程直接继承 CrewAI `Flow` 并启用持久化，
负责管理任务规划、步骤依赖、执行状态、检查点和结果汇总。

```text
自然语言、图像、数据或 JSON/YAML
                  |
          CrewAI 持久化 Flow
                  |
              PGA Crew
       请求解析、校验、任务规划
                  |
    SGA / MCA / CIA / EEA Crews
                           |
                      TPS 与 PES
```

- **PGA**：解析自然语言、图像和数据附件，校验结构化请求并生成带依赖关系
  的执行计划。
- **SGA**：生成 OpenDRIVE、FBX、纹理和 CARLA 导入包，校验道路拓扑，
  并处理 PGA 已验证且经用户确认的农田和树带区域。
- **MCA**：选择已有农机蓝图，生成项目支持的拖拉机参数模板。
- **CIA**：准备纯追踪、PID 和换挡控制，按开关执行贝叶斯参数优化。
- **EEA**：在 CARLA 中执行轨迹跟踪或全流程农业导航实验。
- **TPS、PES**：分别负责轨迹标准化与前进/倒车识别、实验记录与评价，
  二者是 EEA 的下属智能体。CIA 与 EEA 同层，但在闭环实验中也为 EEA
  提供逐帧控制服务。

智能体和任务定义位于：

```text
agri_carla/crewai_runtime/config/agents.yaml
agri_carla/crewai_runtime/config/tasks.yaml
```

每个计划步骤均由 CrewAI 的 `@CrewBase`、`@agent`、`@task` 和 `@crew`
声明创建一个只包含指定智能体和指定项目工具的 Crew。官方事件监听器记录
Flow、Crew、Agent、Task 和 Tool 生命周期。
CrewAI 负责角色执行与工具调用，场景生成、参数配置和 CARLA 控制仍由经过
测试的项目代码完成，模型不能构造任意 Python 调用。

## 2. 主要功能

- 将自然语言、卫星图、轨迹数据和模板整理为经过校验的结构化任务；
- 根据道路中心线、节点、宽度、材质和地理锚点生成农业道路场景；
- 输出 OpenDRIVE、FBX、TGA、预览图和 CARLA 导入包；
- 校验道路端点、连通分量、junction、connection 和 laneLink；
- 根据已确认的场景多边形生成无碰撞农田填充区和可碰撞树带布设计划；
- 生成默认 DF1204 拖拉机参数模板；
- 标准化真实轨迹并识别前进和倒车路段；
- 执行纯追踪横向控制、PID 速度控制和换挡状态管理；
- 通过贝叶斯优化搜索前视距离、速度增益和 PID 参数；
- 执行田内轨迹跟踪和全流程农业导航案例；
- 输出误差指标、路径图、逐帧记录和任务级评价结果；
- 记录智能体状态、进度、错误、产物和检查点，支持查询与断点续跑。

## 3. 项目目录

```text
agri_carla/
  crewai_runtime/   # 唯一顶层 Flow、Crew、工具和 YAML 配置
  execution/        # 白名单领域操作分发器，不具备顶层编排能力
  worker/           # 同机 Python 3.8 CARLA 执行环境
  agents/           # PGA、SGA、MCA、EEA
  controllers/      # CIA、纯追踪、PID、换挡和贝叶斯优化
  trajectory/       # TPS 与轨迹处理
  evaluation/       # PES 与评价指标
  simulation/       # 两类 CARLA 实验
  machinery/        # 拖拉机物理参数
  environment/      # 农田和树带布设
  road_generation/  # 农业道路生成接口
roadrunner_mcp/     # SGA 的 OpenDRIVE、FBX、TGA 底层模块
configs/            # 示例任务、提示词和 JSON Schema
data/               # 论文示例轨迹及 3.1 节五组公开实验结果
scripts/            # 命令行入口
tests/              # 单元测试与架构测试
runtime/            # 运行时自动生成，不纳入版本控制
```

3.1 节五组路径跟踪实验的逐帧记录、指标汇总和诊断图位于
[`data/experiment_results/section_3_1/`](data/experiment_results/section_3_1/README.md)。

## 4. 单台 CARLA 电脑安装

平台只采用同机双环境部署：CrewAI 管理环境使用 Python 3.10-3.13，CARLA
执行环境使用与现有 CARLA 0.9.15 egg 匹配的 Python 3.8。两个环境位于同一台
Linux CARLA 电脑，不需要跨电脑同步文件。

```bash
cd /path/to/CARLA_Agri_MultiAgent_Platform
export CARLA_ROOT="$HOME/carla"
CREWAI_PYTHON=python3.10 bash scripts/setup_crewai_manager.sh
CARLA_PYTHON=python3.8 bash scripts/setup_carla_worker_linux.sh

cp .env.example .env
# 仅在本机编辑 .env，填写真实的模型 API key。

bash scripts/run_same_host_crewai_carla.sh \
  configs/yantai_tracking_experiment_request.json
```

启动脚本先在 `127.0.0.1` 上启动 Python 3.8 Worker，再由 Python 3.10 CrewAI
Flow 调用 EEA。非回环地址会被程序拒绝。Worker 只执行已经确定的领域步骤，
不承担任务规划或顶层调度。安装、运行和故障排查见
[单台 CARLA 电脑部署方案](docs/single_computer_deployment_CN.md)。CrewAI 官方
抽象与项目代码的对应见
[CrewAI 框架实现对应说明](docs/crewai_compliance_CN.md)。

## 5. 任务规划与本地工具

在未连接 CARLA Worker 时，管理端可以完成任务解析、规划、SGA 场景生成、
MCA 参数模板生成等不依赖 CARLA 的工作。下面的命令只进行解析和规划，不会
启动车辆实验：

```bash
python scripts/run_crewai.py \
  --request configs/yantai_tracking_experiment_request.json \
  --plan-only
```

自然语言和附件输入：

```bash
python scripts/run_crewai.py \
  --text "在 YantaiTrackingField 中使用默认拖拉机跟踪给定实测轨迹，并评价跟踪误差。" \
  --data data/trajectory/yantai_quanlujing.csv \
  --request-template configs/existing_scene_tracking_request.yaml \
  --llm-config configs/llm/gpt_5_6_sol.json \
  --plan-only
```

轨迹跟踪和全流程导航需要先启动 CARLA，再使用第 4 节的同机双环境入口。
`--plan-only` 不连接 CARLA。

## 6. 状态、检查点和恢复

```bash
python scripts/run_crewai.py --status TASK_ID
python scripts/run_crewai.py --resume TASK_ID
```

管理端默认保存：

```text
runtime/crewai_runtime.sqlite3
runtime/crewai_flow_states.sqlite3
runtime/runs/<task_id>/structured_request.json
runtime/runs/<task_id>/execution_plan.json
runtime/runs/<task_id>/validation_report.json
runtime/runs/<task_id>/checkpoints/
```

Flow 状态包括 `RECEIVED`、`PLANNING`、`READY`、`RUNNING`、`RETRYING`、
`SUCCEEDED`、`FAILED`、`BLOCKED` 和 `NEEDS_INPUT`。网络或依赖错误可自动
重试；恢复任务时会跳过已有成功检查点的步骤。

## 7. 两个论文案例

下列任务均会进入 EEA，因此必须先启动 CARLA，并使用第 4 节的同机双环境
入口。仅做规划时可在管理环境运行 `scripts/run_crewai.py --plan-only`。

轨迹跟踪：

```bash
bash scripts/run_same_host_crewai_carla.sh \
  configs/yantai_tracking_experiment_request.json
```

使用仓库内烟台轨迹进行单轨迹贝叶斯参数优化：

```bash
bash scripts/run_same_host_crewai_carla.sh \
  configs/yantai_tracking_optimization_request.json
```

全流程农业导航：

```bash
bash scripts/run_same_host_crewai_carla.sh \
  configs/yantai_full_process_agent_request.json
```

CIA 优化变量包括前进/倒车最小前视距离、对应速度增益和 PID 的 `Kp`、
`Ki`、`Kd`。

多轨迹优化与独立测试是可选扩展。参数优化集位于
`data/trajectory/parameter_optimization_set/`，独立测试集位于
`data/trajectory/independent_test_set/`。这两个目录在公开示例中只有说明文件，
加入正式 CSV 后才能运行：

```bash
bash scripts/run_same_host_crewai_carla.sh \
  configs/yantai_tracking_dataset_optimization_request.json
```

独立测试结果不反馈给优化器。若目录中没有 CSV，程序会明确拒绝执行，避免把
空数据集误报为完成。

### 7.1 命令与实际实现文件

CrewAI 是统一入口，因此不再保留绕过 PGA 的独立全流程启动脚本。论文案例的
实现代码仍完整保留，调用关系如下：

| 功能 | 命令入口 | 实际实现 |
|---|---|---|
| CrewAI Flow | `scripts/run_crewai.py` | `agri_carla/crewai_runtime/flow.py` |
| 轨迹跟踪实验 | Flow 调用 EEA | `agri_carla/simulation/tracking_experiment.py` |
| 贝叶斯参数优化 | CIA 在跟踪任务内执行 | `agri_carla/controllers/controller_optimization.py` |
| 全流程导航实验 | Flow 调用 EEA | `agri_carla/simulation/full_process_experiment.py` |
| 场景生成 | Flow 调用 SGA | `agri_carla/road_generation/agent.py` 与 `roadrunner_mcp/realmap.py` |

主要优化输出：

```text
controller_optimization_trials.csv
controller_optimization_convergence.png
best_controller_config.json
controller_optimization_summary.json
independent_test/independent_test_metrics.csv
independent_test/independent_test_comparison.csv
independent_test/independent_test_summary.json
```

## 8. 原始卫星图场景复核

首轮输入使用未标注的原始卫星图，并通过自然语言说明目标道路、农田、树带、
道路宽度、材质、比例尺和地理锚点。PGA 生成道路及环境区域候选，进行第二次
视觉复核和本地几何校验，并输出叠加预览图。只有用户确认候选结果后，SGA
才生成场景文件和植被布设计划。

若首轮候选不符合要求，可补充文字重新规划；确有必要时，第二轮可提供修订图，
其标注采用以下约定：

- 闭合半透明多边形表示农田或草坪填充区域；
- 狭长多边形或带状框选表示树木种植区域或树带；
- 单个手绘点不解释为精确单株树坐标。

草坪区域默认不产生碰撞，树木可配置为产生碰撞。没有 CARLA 时可使用
`dry_run` 生成布设计划和预览图。

仓库不附带第三方卫星影像。用户应使用自有影像或具有再分发许可的数据，并在
命令行通过 `--image /path/to/satellite_image.png` 提供。示例提示词和请求模板
分别位于 `configs/prompts/raw_scene_generation_prompt.example.txt` 与
`configs/raw_scene_generation_template.example.json`。

## 9. 直接工具入口

这些脚本可用于调试单个领域功能，但不承担顶层编排：

```bash
# 轨迹标准化
python scripts/process_trajectory.py \
  --input data/trajectory/yantai_quanlujing.csv \
  --output outputs/trajectories/yantai_quanlujing_standard.csv

# 农机模板
python scripts/generate_machinery_template.py \
  --type tractor \
  --output outputs/templates/tractor_df1204.json

# 农业道路生成
python scripts/generate_road_scene.py \
  --request configs/road_scene_request.yaml \
  --output-dir outputs/road_scene

# 直接调试 CARLA 跟踪执行器（必须在 CARLA Python 环境中运行并已启动模拟器）
python scripts/run_tracking_experiment.py \
  --trajectory data/trajectory/yantai_quanlujing.csv \
  --output-dir results/trajectory_tracking
```

## 10. 测试

```bash
python scripts/audit_release.py
python -m compileall -q agri_carla scripts tests
python -m unittest discover -s tests -v
```

发布目录的保留内容、排除项和上传前检查见
[发布目录说明](docs/release_contents_CN.md)。



## 11. 开源许可证

本项目采用 [MIT License](LICENSE) 开源。
