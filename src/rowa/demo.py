"""Gradio demo: grade an AP Row A thesis with the specialist vs the frontier, side by side.

    pip install -r requirements-demo.txt
    python -m src.rowa.demo                       # specialist (tuned) vs gpt-4o
    python -m src.rowa.demo --no-frontier         # specialist only (no API key needed)

Paste an LEQ/DBQ prompt and a candidate thesis; see each grader's 0/1 decision + reason.
"""

from __future__ import annotations

import argparse
import os

from src.rowa import rubric

EXAMPLES = [
    [rubric.PROMPTS["2023"],
     "The Mongols made Silk Road trade safer, which increased cultural exchange between distant peoples."],
    [rubric.PROMPTS["2023"],
     "The Mongol conquests affected many of the peoples of Eurasia during this time period."],
    [rubric.PROMPTS["2024"],
     "It is beyond dispute that the Mongol conquests profoundly and irrevocably transformed "
     "the lives of countless peoples across Eurasia."],
]


def build_app(specialist, frontier):
    import gradio as gr

    def grade(prompt, thesis):
        system = rubric.grader_system("hardened")
        user = rubric.grader_user(prompt, thesis)
        out = {}
        try:
            r = specialist.grade(system, user)
            out["specialist"] = f"### Specialist (Qwen3-0.6B)\n**{'EARNS' if r.point else 'DENIES'} the point** ({r.point})\n\n{r.reason}"
        except Exception as e:
            out["specialist"] = f"specialist error: {e}"
        if frontier is not None:
            try:
                r = frontier.grade(system, user)
                out["frontier"] = f"### Frontier (gpt-4o, hardened)\n**{'EARNS' if r.point else 'DENIES'} the point** ({r.point})\n\n{r.reason}"
            except Exception as e:
                out["frontier"] = f"frontier error: {e}"
        else:
            out["frontier"] = "_frontier disabled_"
        return out["specialist"], out["frontier"]

    with gr.Blocks(title="AP Row A Thesis Grader") as app:
        gr.Markdown("# AP World History — Row A Thesis Grader\n"
                    "A fine-tuned 0.6B specialist vs. gpt-4o, grading the single thesis point.")
        prompt = gr.Textbox(label="LEQ / DBQ prompt", lines=3, value=rubric.PROMPTS["2023"])
        thesis = gr.Textbox(label="Student thesis", lines=3)
        btn = gr.Button("Grade Row A", variant="primary")
        with gr.Row():
            spec_out = gr.Markdown()
            front_out = gr.Markdown()
        btn.click(grade, [prompt, thesis], [spec_out, front_out])
        gr.Examples(EXAMPLES, [prompt, thesis])
    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", default=os.getenv("TUNED_MODEL", "outputs/rowa-thesis-qlora"))
    ap.add_argument("--base-model", default=os.getenv("BASE_MODEL", "Qwen/Qwen3-0.6B"))
    ap.add_argument("--no-frontier", action="store_true")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    from src.rowa.grader import LocalGrader, FrontierGrader

    specialist = LocalGrader(args.base_model, args.adapter)
    frontier = None if args.no_frontier else FrontierGrader(backend="openai")
    build_app(specialist, frontier).launch(share=args.share)


if __name__ == "__main__":
    main()
