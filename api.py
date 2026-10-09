"""Zero-dependency local API endpoint for the mini-RAG pipeline."""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# Import our existing pipeline logic
from pipeline import (
    load_documents,
    chunk_documents,
    build_index,
    retrieve,
    generate_answers,
)

# Initialize the index in memory when the server starts
print("Loading KB and building index for API...")
kb_dir = Path("kb")
if not kb_dir.exists():
    raise FileNotFoundError("kb/ directory not found. Please create it and add .txt files.")

docs = load_documents(kb_dir)
chunks = chunk_documents(docs, strategy="line")
index = build_index(chunks)
print("Index ready! Starting server on port 8000...")


class RAGRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/answer":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)

            try:
                request_json = json.loads(post_data)
                question = request_json.get("question", "")
                
                if not question:
                    raise ValueError("Missing 'question' in request body")

                # Run the query through our pipeline components
                mock_query = [{"query_id": "API_Q", "question": question}]
                retrieval_results = retrieve(index, chunks, mock_query, top_k_count=3)
                answers = generate_answers(retrieval_results)
                
                final_answer = answers[0]
                
                # Format response exactly as requested by the prompt
                response_data = {
                    "answer_label": final_answer["answer_label"],
                    "answer": final_answer["answer"],
                    "citations": final_answer["citations"]
                }
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode("utf-8"))

            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                error_resp = {"error": str(e)}
                self.wfile.write(json.dumps(error_resp).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def run_server(port=8000):
    server_address = ("", port)
    httpd = HTTPServer(server_address, RAGRequestHandler)
    print(f"API listening at http://localhost:{port}/answer")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()

if __name__ == "__main__":
    run_server()