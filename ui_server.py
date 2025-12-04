


from flask import Flask, request, render_template_string
from white_client import ask_white_agent

app = Flask(__name__)

HTML = """
<!doctype html>
<html>
  <body style="max-width:800px;margin:40px auto;font-family:sans-serif;">
    <h2>AIPolicyAgentBench – Simple UI</h2>
    <form method="POST">
        <textarea name="question" style="width:100%;height:140px;">{{ q }}</textarea>
        <br><br>
        <button type="submit">Ask White Agent</button>
    </form>

    {% if answer %}
      <h3>Response</h3>
      <pre style="white-space:pre-wrap;background:#f6f6f6;padding:15px;border-radius:8px;">{{ answer }}</pre>
    {% endif %}
  </body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    answer = None
    q = ""
    if request.method == "POST":
        q = request.form.get("question", "")
        answer = ask_white_agent(q)
    return render_template_string(HTML, answer=answer, q=q)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


