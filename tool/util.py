import os


try:
    # Running as a Python script (inside src/)
    this_file = os.path.abspath(__file__)
    src_root = os.path.dirname(this_file)
    project_root = os.path.dirname(src_root)

except NameError:
    # Running inside Jupyter
    cwd = os.getcwd()

    if cwd.endswith("notebooks"):
        src_root = os.path.abspath(os.path.join(cwd, ".."))
        project_root = os.path.dirname(src_root)

    elif os.path.basename(cwd) == "src":
        src_root = cwd
        project_root = os.path.dirname(src_root)

    else:
        # Running directly from project root
        project_root = cwd
        src_root = os.path.join(project_root, "src")

# ===== Canonical project paths =====
data_root    = os.path.join(project_root, "data")
prompts_root = os.path.join(project_root, "prompts")
utils_root   = os.path.join(project_root, "utils")
results_root = os.path.join(project_root, "results")
src_root     = os.path.join(project_root, "src")

print(
    f"📂 Project root : {project_root}\n"
    f"📂 Source root  : {src_root}\n"
    f"📂 Data root    : {data_root}\n"
    f"📂 Prompts root : {prompts_root}\n"
    f"📂 Utils root   : {utils_root}\n"
    f"📂 Results root : {results_root}"
)