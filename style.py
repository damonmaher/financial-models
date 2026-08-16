"""
style.py

Design tokens + CSS injection for the "quant-math panel" dark aesthetic used
across this suite of tools: a near-black terminal/ledger surface, monospace
data typography, and a single phosphor-teal accent used for anything
computed (as opposed to input).

Call apply_theme() once, near the top of main.py, before building the page.
"""

from nicegui import ui

# ---- color tokens -----------------------------------------------------
BG_VOID = "#0a0d12"
BG_PANEL = "#10141c"
BG_PANEL_RAISED = "#141924"
BORDER = "#232937"
BORDER_SOFT = "#1a1f2b"

TEXT_PRIMARY = "#e8ecf1"
TEXT_MUTED = "#6b7686"
TEXT_DIM = "#454c5c"

ACCENT = "#5eead4"        # phosphor teal - computed values, focus states
ACCENT_DIM = "#2c8577"
AMBER = "#f5a623"         # inputs / user-authored views
POSITIVE = "#34d399"      # long / gain
NEGATIVE = "#f87171"      # short / loss

FONT_DISPLAY = "'Space Grotesk', sans-serif"
FONT_BODY = "'Inter', sans-serif"
FONT_MONO = "'JetBrains Mono', monospace"


def apply_theme() -> None:
    ui.dark_mode().enable()

    ui.add_head_html(f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-void: {BG_VOID};
            --bg-panel: {BG_PANEL};
            --bg-panel-raised: {BG_PANEL_RAISED};
            --border: {BORDER};
            --border-soft: {BORDER_SOFT};
            --text-primary: {TEXT_PRIMARY};
            --text-muted: {TEXT_MUTED};
            --text-dim: {TEXT_DIM};
            --accent: {ACCENT};
            --accent-dim: {ACCENT_DIM};
            --amber: {AMBER};
            --positive: {POSITIVE};
            --negative: {NEGATIVE};
        }}

        body, .q-page, .nicegui-content {{
            background-color: var(--bg-void) !important;
            background-image:
                linear-gradient(var(--border-soft) 1px, transparent 1px),
                linear-gradient(90deg, var(--border-soft) 1px, transparent 1px);
            background-size: 34px 34px;
            background-attachment: fixed;
            color: var(--text-primary);
            font-family: {FONT_BODY};
        }}

        .qp-title {{
            font-family: {FONT_DISPLAY};
            font-weight: 700;
            letter-spacing: 0.04em;
            color: var(--text-primary);
        }}

        .qp-eyebrow {{
            font-family: {FONT_MONO};
            font-size: 0.72rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: var(--accent);
        }}

        .qp-mono {{
            font-family: {FONT_MONO} !important;
        }}

        .qp-muted {{
            color: var(--text-muted) !important;
        }}

        .qp-panel {{
            background: var(--bg-panel) !important;
            border: 1px solid var(--border) !important;
            border-radius: 6px !important;
            box-shadow: 0 0 0 1px rgba(94, 234, 212, 0.03), 0 12px 28px -20px rgba(0,0,0,0.8);
        }}

        .qp-panel-header {{
            font-family: {FONT_MONO};
            font-size: 0.72rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--text-muted);
            border-bottom: 1px solid var(--border);
            padding: 10px 16px;
        }}

        .qp-divider {{
            border-color: var(--border) !important;
        }}

        /* inputs */
        .q-field__control {{
            background: var(--bg-panel-raised) !important;
            border-radius: 4px !important;
        }}
        .q-field--outlined .q-field__control:before {{
            border-color: var(--border) !important;
        }}
        .q-field__native, .q-field__input {{
            color: var(--text-primary) !important;
            font-family: {FONT_MONO} !important;
        }}
        .q-field--focused .q-field__control:before {{
            border-color: var(--accent) !important;
        }}
        .q-field__label {{
            font-family: {FONT_BODY} !important;
            color: var(--text-muted) !important;
        }}

        /* buttons */
        .qp-btn-primary {{
            background: var(--accent) !important;
            color: #06110f !important;
            font-family: {FONT_MONO} !important;
            font-weight: 700 !important;
            letter-spacing: 0.06em;
        }}
        .qp-btn-ghost {{
            background: transparent !important;
            color: var(--text-muted) !important;
            border: 1px solid var(--border) !important;
            font-family: {FONT_MONO} !important;
        }}
        .qp-btn-amber {{
            background: transparent !important;
            color: var(--amber) !important;
            border: 1px dashed var(--amber) !important;
            font-family: {FONT_MONO} !important;
        }}

        /* chips for tickers */
        .qp-ticker-chip {{
            font-family: {FONT_MONO} !important;
            background: var(--bg-panel-raised) !important;
            border: 1px solid var(--border) !important;
            color: var(--accent) !important;
            font-weight: 600;
        }}

        /* table */
        .qp-table thead tr {{
            background: var(--bg-panel-raised) !important;
        }}
        .qp-table th {{
            font-family: {FONT_MONO} !important;
            font-size: 0.7rem !important;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--text-muted) !important;
            border-bottom: 1px solid var(--border) !important;
        }}
        .qp-table td {{
            font-family: {FONT_MONO} !important;
            color: var(--text-primary) !important;
            border-bottom: 1px solid var(--border-soft) !important;
        }}
        .qp-table .q-table__container {{
            background: var(--bg-panel) !important;
        }}

        .qp-tabs .q-tab {{
            font-family: {FONT_MONO} !important;
            font-size: 0.75rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--text-muted) !important;
        }}
        .qp-tabs .q-tab--active {{
            color: var(--accent) !important;
        }}
        .qp-tab-panels {{
            background: transparent !important;
        }}

        ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
        ::-webkit-scrollbar-track {{ background: var(--bg-void); }}
        ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 4px; }}
    </style>
    """)