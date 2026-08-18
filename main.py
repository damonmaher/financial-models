import ast
import os
import sys
import threading
from pathlib import Path

from nicegui import ui


# ============================================================
# MODEL EMBEDDING
# ============================================================
#
# The three model applications in math_code/ were originally
# written as standalone NiceGUI applications.
#
# We execute those ORIGINAL files inside the appropriate
# NiceGUI page rather than recreating their UI here.
#
# The only thing removed from the source at runtime is the
# model's own ui.run(...) call, because the ROOT application
# owns the one Render web server.
#
# No model math is changed.
# No model UI is changed.
# ============================================================

_REPO_ROOT = Path(__file__).resolve().parent

_MODEL_PATHS = {
    "bl": _REPO_ROOT / "math_code" / "bl" / "main.py",
    "hmm": _REPO_ROOT / "math_code" / "hmm" / "main.py",
    "hsv": _REPO_ROOT / "math_code" / "hsv" / "main.py",
}

_MODEL_LOAD_LOCK = threading.Lock()


def _remove_ui_run_calls(source: str, filename: str) -> object:
    """
    Parse the ORIGINAL model source and remove only calls to
    ui.run(...).

    Everything else — including the model's UI construction,
    callbacks, plots, calculations, styling, etc. — is left
    untouched.
    """
    tree = ast.parse(source, filename=filename)

    class RemoveUiRun(ast.NodeTransformer):
        def visit_Expr(self, node):
            node = self.generic_visit(node)

            if isinstance(node, ast.Expr):
                call = node.value

                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "ui"
                    and call.func.attr == "run"
                ):
                    return ast.Pass()

            return node

    tree = RemoveUiRun().visit(tree)
    ast.fix_missing_locations(tree)
    return compile(tree, filename, "exec")


def _run_original_model(model_name: str):
    """
    Execute one of the ORIGINAL standalone model applications
    inside the current NiceGUI page.

    The model's own directory is temporarily placed first on
    sys.path because the original model files use imports such
    as:

        import data_handler
        import black_litterman
        import optimizer

    or:

        import set_params
        import learn
        import hmm

    We also isolate those imports so HMM and HSV do not
    accidentally share modules with the same names.
    """

    model_path = _MODEL_PATHS[model_name]
    model_dir = model_path.parent

    source = model_path.read_text(encoding="utf-8")
    code = _remove_ui_run_calls(source, str(model_path))

    # These are the local modules used by the standalone apps.
    #
    # They intentionally get isolated between model executions
    # because HMM and HSV both have modules named set_params,
    # while HMM/BL/etc. have other local module names.
    local_module_names = {
        "bl": {
            "data_handler",
            "black_litterman",
            "optimizer",
            "style",
        },
        "hmm": {
            "set_params",
            "learn",
            "hmm",
        },
        "hsv": {
            "set_params",
            "bs",
            "fft",
        },
    }[model_name]

    with _MODEL_LOAD_LOCK:
        old_path = list(sys.path)

        # Save currently loaded modules with the same names.
        saved_modules = {
            name: sys.modules.get(name)
            for name in local_module_names
        }

        try:
            # The model's own directory must come first so its
            # original imports resolve exactly as they did when
            # the model was run standalone.
            sys.path.insert(0, str(model_dir))

            # Remove potentially conflicting modules.
            for name in local_module_names:
                sys.modules.pop(name, None)

            # Give the executed file a non-main name so guarded
            # standalone blocks do not call ui.run().
            namespace = {
                "__name__": f"_embedded_{model_name}_model",
                "__file__": str(model_path),
                "__package__": None,
                "__builtins__": __builtins__,
            }

            exec(code, namespace, namespace)

        finally:
            sys.path[:] = old_path

            # Restore whatever module names existed before.
            for name, module in saved_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module


# ============================================================
# HOME
# ============================================================

@ui.page("/")
def home():
    ui.label("Mahini & Maher Live Models").classes(
        "text-2xl font-bold mb-4"
    )

    ui.link(
        "Black-Litterman (BL)",
        "/bl",
    ).classes("block mb-2 text-blue-500")

    ui.link(
        "Hidden Markov Model (HMM)",
        "/hmm",
    ).classes("block mb-2 text-blue-500")

    ui.link(
        "Heston Stochastic Volatility (HSV)",
        "/hsv",
    ).classes("block mb-2 text-blue-500")


# ============================================================
# BLACK-LITTERMAN
# ============================================================

@ui.page("/bl")
def bl_page():
    _run_original_model("bl")


# ============================================================
# HIDDEN MARKOV MODEL
# ============================================================

@ui.page("/hmm")
def hmm_page():
    _run_original_model("hmm")


# ============================================================
# HESTON STOCHASTIC VOLATILITY
# ============================================================

@ui.page("/hsv")
def hsv_page():
    _run_original_model("hsv")


# ============================================================
# RENDER SERVER
# ============================================================

port = int(os.environ.get("PORT", 10000))

ui.run(
    host="0.0.0.0",
    port=port,
    reload=False,
)
