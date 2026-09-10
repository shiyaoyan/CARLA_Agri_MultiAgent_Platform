# 同机双环境部署方案

## 1. 部署边界

平台只支持部署在同一台 Linux CARLA 工作站上的两个隔离 Python 环境：

- `.venv-crewai`：Python 3.10-3.13，运行 CrewAI Flow、PGA、SGA、MCA 和 CIA；
- `.venv-carla`：Python 3.8，加载 CARLA 0.9.15 egg 并执行 EEA 仿真实验。

两个环境共享同一个项目目录，通过仅监听 `127.0.0.1` 的 HTTP Worker 交换
已经校验的结构化任务和执行结果。项目不提供单环境模式，也不支持跨电脑 Worker。
CARLA 服务端仍是独立仿真程序，不受两个 Python 虚拟环境的划分影响。

## 2. 安装

```bash
cd /path/to/CARLA_Agri_MultiAgent_Platform
export CARLA_ROOT="$HOME/carla"

CREWAI_PYTHON=python3.10 bash scripts/setup_crewai_manager.sh
CARLA_PYTHON=python3.8 bash scripts/setup_carla_worker_linux.sh

cp .env.example .env
```

在 `.env` 中填写真实的 `OPENAI_API_KEY`。Worker 地址保留为：

```text
AGRI_CARLA_WORKER_URL=http://127.0.0.1:8765
AGRI_CARLA_WORKER_AGENTS=EEA
```


## 3. 运行

先启动 CARLA 服务端并加载任务所需地图，然后运行：

```bash
bash scripts/run_same_host_crewai_carla.sh \
  configs/yantai_tracking_experiment_request.json
```

该脚本依次完成：

1. 检查 `.venv-crewai`、`.venv-carla` 和 CARLA Python 3.8 egg；
2. 在 `127.0.0.1:8765` 启动 CARLA Worker；
3. 使用随机令牌或已有令牌完成本机接口认证；
4. 完成 Worker 健康检查；
5. 在管理环境启动 CrewAI Flow；
6. 将 EEA 步骤交给 Python 3.8 Worker；
7. 任务结束后关闭本次脚本启动的 Worker。

启动器也可以直接接收 `run_crewai.py` 的自然语言参数。例如：

```bash
bash scripts/run_same_host_crewai_carla.sh \
  --text-file configs/prompts/yantai_full_process_prompt.txt \
  --request-template configs/yantai_full_process_agent_request.json \
  --llm-config configs/llm/gpt_5_6_sol.json
```

默认只有 EEA 进入 CARLA 环境。若某个 SGA 操作需要直接连接已经导入的
CARLA 地图，可在请求的 `execution.worker_agents` 中加入 `SGA`，但地址仍必须
是本机回环地址。

## 4. 只进行任务规划

不运行 CARLA 实验时，可在管理环境执行：

```bash
source .venv-crewai/bin/activate
python scripts/run_crewai.py \
  --request configs/yantai_tracking_experiment_request.json \
  --plan-only
```

该命令可以完成 PGA 解析、校验和执行计划生成，不会调用 Python 3.8 Worker。

## 5. 状态与恢复

```bash
source .venv-crewai/bin/activate
python scripts/run_crewai.py --status TASK_ID
python scripts/run_crewai.py --resume TASK_ID
```

如果恢复任务仍需执行 EEA，应先在另一个终端使用 Python 3.8 环境启动本机
Worker，或者重新通过同机启动脚本提交完整请求。

## 6. 验收检查

```bash
.venv-crewai/bin/python -c "import crewai, agri_carla; print(crewai.__version__)"

CARLA_EGG="$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.8-linux-x86_64.egg"
PYTHONPATH="$PWD:$CARLA_ROOT/PythonAPI/carla:$CARLA_EGG" \
  .venv-carla/bin/python -c \
  "import carla; from agents.navigation.behavior_agent import BehaviorAgent; print('CARLA worker OK')"
```

完整验收还应包括：连接 CARLA、读取地图、生成拖拉机、执行控制指令、完成
轨迹跟踪案例并生成误差表，以及完成全流程导航阶段切换。

