# CrewAI 框架实现对应说明

本项目仅使用一个顶层工作流 `AgriculturalSimulationFlow`。
## 官方抽象与项目实现

| CrewAI 抽象 | 项目实现 | 作用 |
|---|---|---|
| `Flow[State]` | `AgriculturalSimulationFlow` | 负责规划、顺序执行、重试、结束和恢复 |
| Pydantic Flow state | `AgriculturalFlowState` | 保存请求、计划、当前步骤、结果和错误状态 |
| `@start` / `@listen` | `plan_task` / `execute_plan` / `finalize_task` | 定义唯一的顶层执行流程 |
| `@persist(SQLiteFlowPersistence(...))` | `AgriculturalSimulationFlow` 类装饰器 | 将 Flow 状态保存到项目 `runtime/` 并按相同 state id 恢复 |
| `@CrewBase` | 五个功能 Crew 类 | 使用声明式 CrewAI 项目结构 |
| `@agent` | PGA、SGA、MCA、CIA、EEA | 定义五个论文中的功能智能体 |
| `@task` | `config/tasks.yaml` 对应任务 | 约束每个智能体的一次工具调用与输出 |
| `@crew` / `Crew` | 每个计划步骤的单智能体 Crew | 由 Flow 调用并执行当前步骤 |
| `Agent` | `config/agents.yaml` 角色配置 | 定义角色、目标、背景和迭代上限 |
| `BaseTool` | `PlatformAgentTool` | 只开放通过白名单校验的项目能力 |
| Pydantic tool schema | `PlatformToolInput` | 使用空输入 schema；任务上下文由 Flow 预绑定，模型无法改写 |
| `BaseEventListener` | `AgriculturalCrewAIEventListener` | 接收 Flow、Crew、Agent、Task 和 Tool 事件 |

## 当前运行管理职责

| 管理职责 | CrewAI 架构中的实现 |
|---|---|
| 任务识别与计划 | PGA Crew 和 `plan_task` |
| 智能体调用顺序 | PGA 生成依赖计划，Flow 的 `execute_plan` 执行 |
| 智能体运行状态 | Flow state、官方事件监听器和运行账本 |
| 错误反馈 | CrewAI 失败事件、Flow 异常分类和最终失败报告 |
| 自动重试 | Flow 对瞬时连接或依赖错误的限次重试 |
| 检查点和断点恢复 | `@persist` Flow state 与成功领域步骤检查点 |
| 执行记录 | `ApplicationRunLedger` 中的 SQLite 审计记录 |
| 产物登记 | 运行账本记录文件路径、类型和 SHA-256 摘要 |
| 最终报告 | `finalize_task` 汇总 Flow 结果并写入报告 |

`ApplicationRunLedger` 不判断任务类型、不生成步骤、不选择智能体，也不执行
重试，因此不是第二套编排器。CrewAI 的持久化状态用于恢复工作流；项目账本
额外保存论文实验所需的领域检查点、产物摘要和可人工审查的 JSON 文件。
CrewAI Flow 自动创建的官方 Memory 存放在 `runtime/crewai_storage/`，不使用
系统级用户数据目录。

## CrewAI 与 CARLA 执行边界

平台采用同机双环境：CrewAI 1.15.12 运行于 Python 3.10 管理环境，原 CARLA
0.9.15 egg 运行于 Python 3.8 执行环境，两者通过仅监听 `127.0.0.1` 的带令牌
接口通信。顶层任务规划、状态和恢复始终由唯一的 CrewAI Flow 管理。

Worker 中的 `DomainExecutionService` 只把已经确定的 `agent/operation`
映射到现有农业功能，不包含计划、路由或调度逻辑。

## 版本约束

- CrewAI：`1.15.12`
- CrewAI 管理环境：Python `3.10-3.13`
- CARLA Worker 环境：Python `3.8`
- 部署方式：同一台 Linux 工作站上的 `.venv-crewai` 与 `.venv-carla`
