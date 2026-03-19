import os
import subprocess
from pathlib import Path

def collect_code_and_tree(
    root_dir: str = ".",
    output_file: str = "collected_code.txt",
    exclude_dirs: set = None,
    extensions: tuple = (".py", ".yaml", ".yml")
):
    if exclude_dirs is None:
        exclude_dirs = {"venv", "__pycache__", ".git", "node_modules"}

    # Определяем имя текущего скрипта для исключения
    current_script = Path(__file__).name  # Например: "collect_code.py"
    output_filename = Path(output_file).name  # Чтобы не включать сам результат

    TREE_PATH = "/usr/bin/tree"  # ← Убедись, что путь актуален (проверь через `which tree`)
    output_path = Path(output_file)
    root_path = Path(root_dir).resolve()

    # === ШАГ 1: tree ===
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== Структура проекта ===\n")
        try:
            exclude_pattern = "|".join(exclude_dirs)
            result = subprocess.run(
                [TREE_PATH, "-I", exclude_pattern, "-n", "--dirsfirst"],
                cwd=root_path,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                f.seek(0, 2)
                f.write(f"\n<tree завершился с кодом {result.returncode}>\n")
        except Exception as e:
            f.write(f"<Ошибка запуска tree: {e}>\n")
        f.write("\n=== Содержимое файлов ===\n\n")

    # === ШАГ 2: сбор файлов с исключением скрипта и выходного файла ===
    with open(output_path, "a", encoding="utf-8") as out_f:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
            for filename in sorted(filenames):
                # Пропускаем сам скрипт и файл вывода
                if filename == current_script or filename == output_filename:
                    continue
                if filename.endswith(extensions):
                    filepath = os.path.join(dirpath, filename)
                    rel_path = os.path.relpath(filepath, root_dir)
                    try:
                        with open(filepath, "r", encoding="utf-8") as src_f:
                            content = src_f.read()
                        out_f.write(f"***\n{rel_path}\n***\n{content}\n***\n\n")
                    except Exception as e:
                        out_f.write(f"***\n{rel_path}\n***\n<Ошибка чтения: {e}>\n***\n\n")

    print(f"✅ Готово: {output_path.resolve()}")

if __name__ == "__main__":
    collect_code_and_tree()