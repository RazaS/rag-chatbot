import os
from flask import Flask, request, render_template_string
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import google.generativeai as genai
from dotenv import load_dotenv  # ✅ This was missing


# === CONFIG ===
INDEX_NAME = "guideline-rag"

load_dotenv()  # Load variables from .env

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# === Init Services ===
genai.configure(api_key=GOOGLE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# === Flask UI ===
from flask import session
app = Flask(__name__)
app.secret_key = os.urandom(24)  # required for session

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
  <head>
    <title>Guideline Chatbot</title>
    <style>
      body { font-family: sans-serif; max-width: 700px; margin: auto; }
      .chat-box { background: #f0f0f0; padding: 1em; margin: 1em 0; border-radius: 8px; }
      .user { font-weight: bold; color: #0074d9; }
      .bot { white-space: pre-wrap; }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  </head>
  <body>
    <h2>Guideline Chatbot</h2>
    <form method="POST">
      <input name="query" style="width: 400px;" required autofocus>
      <input type="submit" value="Ask">
    </form>
    <div id="chat-history">
      {% for pair in history %}
        <div class="chat-box">
          <div class="user">You:</div>
          <div>{{ pair.query }}</div>
          <br>
          <div class="user">Bot:</div>
          <div class="bot" id="bot-{{ loop.index }}"></div>
          <script>
            document.getElementById("bot-{{ loop.index }}").innerHTML = marked.parse(`{{ pair.response | tojson | safe }}`);
          </script>
        </div>
      {% endfor %}
    </div>
  </body>
</html>
"""

def retrieve_context(query, top_k=5):
    q_emb = embedder.encode([query])[0].tolist()
    results = index.query(vector=q_emb, top_k=top_k, include_metadata=True)
    return "\n\n".join([m["metadata"]["text"] for m in results["matches"]])

@app.route("/", methods=["GET", "POST"])
def chat():
    if "history" not in session:
        session["history"] = []
    response = None
    if request.method == "POST":
        query = request.form["query"]
        context = retrieve_context(query)
        model = genai.GenerativeModel("gemini-1.5-flash")
        result = model.generate_content(f"Use this context to answer the question:\n\n{context}\n\nQuestion: {query}")
        response = result.text
        session["history"].append({"query": query, "response": response})
        session.modified = True
    return render_template_string(HTML_TEMPLATE, response=response, history=session.get("history", []))

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))  # fallback to 8080 locally
    app.run(host="0.0.0.0", port=port)