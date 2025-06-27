from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

MODEL_DIR = os.getenv("MODEL_DIR", "model")
device_index = int(os.getenv("CUDA_DEVICE_INDEX", "-1"))

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
if tokenizer.pad_token is None:     
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device_index,
)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    code_output = ""
    if request.method == "POST":
        prompt = request.form.get("prompt", "").strip()
        if prompt:
            prefix = f"# Instruction: {prompt}\n```python\n"
            try:
                response_text = generator(
                    prefix,
                    max_length=256,
                    do_sample=True,
                    top_p=0.95,
                    temperature=0.6,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id,
                )[0]["generated_text"]

                generated = response_text[len(prefix):]

                generated = generated.split("```")[0]

                plower = prompt.lower()
                cleaned_lines = [
                    ln for ln in generated.splitlines()
                    if ln.strip() and plower not in ln.lower()
                ]
                code_output = "\n".join(cleaned_lines).strip()
            except Exception as e:
                code_output = f"Error: {e}"
    return render_template("index.html", code_output=code_output)

if __name__ == "__main__":
    app.run(debug=True, threaded=True)