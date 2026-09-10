# CrewAI Agent Architecture

The platform has one top-level workflow: `AgriculturalSimulationFlow`. It is a
persisted CrewAI `Flow` that owns planning, dependency-aware execution, state,
checkpoints and result aggregation.

## Runtime Structure

```text
Natural-language request, images, data, or JSON/YAML
                         |
              CrewAI persisted Flow
                         |
                PGA Crew and tool
          request parsing, validation, plan
                         |
          dependency-aware Flow execution
               /         |         \
        SGA Crew     MCA Crew     CIA Crew
               \         |         /
                        EEA Crew
                       /         \
                    TPS          PES
```

PGA, SGA, MCA, CIA and EEA are peer CrewAI agents. Each planned step creates
a Crew containing exactly one selected agent and one allow-listed platform
tool. Tool output is returned as the task result, so the model cannot replace
domain output with invented values. TPS and PES remain subordinate services
used only by EEA. CIA remains a peer agent, although EEA also calls its
frame-level controller service while a tracking experiment is running.

Agent and task definitions are stored in:

```text
agri_carla/crewai_runtime/config/agents.yaml
agri_carla/crewai_runtime/config/tasks.yaml
```

## Flow Responsibilities

`agri_carla.crewai_runtime.flow.AgriculturalSimulationFlow` provides:

- a Pydantic structured state;
- PGA planning through a CrewAI Crew;
- dependency-aware step execution;
- explicit `RECEIVED`, `PLANNING`, `READY`, `RUNNING`, `RETRYING`,
  `SUCCEEDED`, `FAILED`, `BLOCKED` and `NEEDS_INPUT` states;
- retry of transient connection and dependency failures;
- a checkpoint after every successful top-level step;
- resume without repeating successful steps;
- event, progress, error and artifact records;
- a stable final result and optional `flow_report`.

`ApplicationRunLedger` stores the project-specific audit trail and domain checkpoints in
SQLite. CrewAI persistence stores Flow state. The store does not decide which
agent runs next and is therefore not a second orchestrator.

Default runtime paths are:

```text
runtime/crewai_runtime.sqlite3
runtime/runs/<task_id>/structured_request.json
runtime/runs/<task_id>/execution_plan.json
runtime/runs/<task_id>/validation_report.json
runtime/runs/<task_id>/checkpoints/
```

## Agent Responsibilities

### PGA

`PlanGenerationAgent` accepts a verified JSON/YAML request or natural-language,
an original satellite image and data attachments. On the first pass, the
multimodal parser uses the image and spatial description to create candidate
road, farmland and tree-belt geometry. A second visual call verifies the
candidate and deterministic validation checks fields, paths and topology. PGA
writes an overlay preview and marks the request as awaiting confirmation. A
rejected candidate may be replanned with clearer text or an optional correction
image; SGA cannot run before explicit confirmation. PGA returns the validated
request and an allow-listed, dependency-aware plan. It does not execute
model-generated Python code.

### SGA

`SceneGenerationAgent` generates OpenDRIVE topology, FBX road
meshes, textures, previews and CARLA import bundles. It validates road
connectivity and repairs only gaps that fall within configured limits. It also
turns PGA-validated and user-confirmed farmland and tree-belt polygons into
vegetation plans; it does not reinterpret the satellite image.

### MCA

`MachineryConfigurationAgent` selects an existing machinery blueprint and
exports a supported parameter template. It does not alter CARLA or Unreal
Engine source assets.

### CIA

`ControllerInterfaceAgent` prepares direction-aware pure-pursuit control, PID
speed control and gear-state settings. Its optional Bayesian optimization
searches controller parameters through repeated EEA experiments and locks the
best observed configuration before independent testing.

### EEA

`ExperimentExecutionAgent` runs trajectory tracking or full-process
navigation in CARLA. It calls TPS for trajectory standardization and
forward/reverse recognition, CIA for controller execution, and PES for
tracking or task-level evaluation.

## Same-host management and CARLA environments

The complete platform runs on one Linux CARLA computer. The CrewAI management
environment performs PGA, SGA, MCA and CIA preparation. EEA is submitted over
`127.0.0.1` to an isolated Python 3.8 CARLA Worker on the same computer. Worker
events and artifacts are replayed into the management task record.

The worker contains only `DomainExecutionService`; it has no planning or
top-level orchestration API. Consequently, the dependency-isolated deployment
still has one workflow owner: the CrewAI Flow in the management environment.
