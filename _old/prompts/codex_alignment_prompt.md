# Prompt for Codex: Fix Status Text Alignment

Use this prompt to instruct Codex to repair the overlapping status text in the Gradio UI. The goal is to align the "Preparing model input…", "Elapsed: 00:00.3", and "ETA: calculating…" labels vertically so they no longer overlap.

---

**Prompt**

> You are working in the rag-sandbox repository on a Gradio-based interface. The status bar currently shows three labels: "Preparing model input…", "Elapsed: 00:00.3", and "ETA: calculating…". They overlap and are misaligned vertically. Update the layout so these texts sit on a single horizontal row with consistent vertical centering and spacing. Use flexbox (or CSS grid) in the relevant container (see `app/assets/custom.css` and the corresponding status component) to align items center vertically and space them evenly. Ensure the design remains responsive and accessible.
> 
> While adjusting the layout, keep verbose logging enabled around any logic that controls the timer/status updates so debugging remains easy. Document any CSS or component changes with concise comments that follow international programming standards for clarity.
> 
> After updating, include a short note summarizing what you changed and why.

---

**Usage Notes**
- Point Codex to the CSS definitions in `app/assets/custom.css` and any status-bar markup so it can align the labels with flex or grid.
- Ask Codex to preserve existing colors, padding, and border styles while only changing alignment and spacing rules.
- Remind Codex to keep verbose logging intact for troubleshooting timer/status rendering.
