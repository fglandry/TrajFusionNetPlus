import subprocess
import json
import tempfile
import os

def run_and_capture_model_path(command: list):
    """
    captured_output, model_path = [], ""
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
        for line in process.stdout:
            print(line, end="")
            captured_output.append(line)
    prefix = "Model saved under: "
    for l in reversed(captured_output):
        if prefix in l:
            model_path = l[len(prefix):].strip()
    return model_path
    """
    temp_files = []
    try:
        cmd = []
        for arg in command:
            if isinstance(arg, dict):
                tf = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w")
                json.dump(arg, tf)
                tf.close()
                temp_files.append(tf.name)
                cmd.append(tf.name)
            else:
                cmd.append(str(arg))

        captured_output, model_path = [], ""
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
            for line in process.stdout:
                print(line, end="")
                captured_output.append(line)
        prefix = "Model saved under: "
        for l in reversed(captured_output):
            if prefix in l:
                model_path = l[len(prefix):].strip()
        return model_path
    finally:
        for f in temp_files:
            try:
                os.remove(f)
            except Exception:
                pass
