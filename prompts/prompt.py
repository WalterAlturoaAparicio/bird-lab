from pathlib import Path
from datetime import datetime


def load_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def summarize_agents(agents_content: str) -> str:
    """
    Reduce ruido del AGENTS.md para Codex.
    Aquí puedes mejorar con IA más adelante.
    """
    sections_to_keep = [
        "## Core Objectives",
        "## Pipeline Stages",
        "## Design Principles",
        "## Constraints & Rules",
        "## Guidelines for AI Assistants"
    ]

    lines = agents_content.splitlines()
    result = []
    capture = False

    for line in lines:
        if any(section in line for section in sections_to_keep):
            capture = True
            result.append(line)
            continue

        # detener captura cuando empieza otra sección
        if capture and line.startswith("## "):
            capture = False

        if capture:
            result.append(line)

    return "\n".join(result)


def build_prompt(
    agents_path: str,
    task_path: str,
    template_path: str,
    output_dir: str = "runs"
) -> str:

    agents_raw = load_file(agents_path)
    task_raw = load_file(task_path)
    template = load_file(template_path)

    agents_summary = summarize_agents(agents_raw)

    prompt = template.format(
        agents=agents_summary,
        task=task_raw
    )

    # Guardar para trazabilidad
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    prompt_file = Path(output_dir) / f"{timestamp}_prompt.txt"

    prompt_file.write_text(prompt, encoding="utf-8")

    return prompt