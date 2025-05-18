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
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
  <body>
    <h2>Guideline Chatbot</h2>
    <form method="POST">
      <input name="query" style="width: 400px;" required autofocus>
      <input type="submit" value="Ask">
    </form>
    {% if response %}
      <h3>Answer:</h3>
      <p>{{ response }}</p>
    {% endif %}
  </body>
</html>
"""

def retrieve_context(query, top_k=5):
    q_emb = embedder.encode([query])[0].tolist()
    results = index.query(vector=q_emb, top_k=top_k, include_metadata=True)
    return "\n\n".join([m["metadata"]["text"] for m in results["matches"]])

@app.route("/", methods=["GET", "POST"])
def chat():
    response = None
    if request.method == "POST":
        query = request.form["query"]
        context = retrieve_context(query)
        model = genai.GenerativeModel("gemini-1.5-flash")
        result = model.generate_content(f"Use this context to answer the question:\n\n{context}\n\nQuestion: {query}")
        response = result.text
    return render_template_string(HTML_TEMPLATE, response=response)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))  # fallback to 8080 locally
    app.run(host="0.0.0.0", port=port)
    