from flask import Flask, request, Response, jsonify, stream_with_context
from flask_cors import CORS
import json
import os
from .api import CopilotAPI

app = Flask(__name__)
CORS(app)

TOKENS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tokens.json")

api_instance = None

def get_api():
    """Load tokens from file and initialize the CopilotAPI instance"""
    global api_instance
    
    try:
        with open(TOKENS_FILE, "r") as f:
            tokens = json.load(f)
            
        gh_token = tokens.get("gh_token")
        vscode_machine = tokens.get("vscode_machine")
        vscode_session = tokens.get("vscode_session")
        
        if not all([gh_token, vscode_machine, vscode_session]):
            raise ValueError("Missing required tokens in tokens.json")
            
        if api_instance is None:
            api_instance = CopilotAPI(gh_token, vscode_machine, vscode_session)
        
        return api_instance
    
    except FileNotFoundError:
        raise FileNotFoundError(f"tokens.json file not found at {TOKENS_FILE}")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON in tokens.json file")

@app.route("/models", methods=["GET"])
def models():
    """Endpoint for fetching available models"""
    try:
        api = get_api()
        models_list = api.models()
        return jsonify({"data": [model.to_dict() for model in models_list]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat():
    """Endpoint for chat completions with streaming support"""
    try:
        api = get_api()
        body = request.json
        
        stream = body.get("stream", False)
        
        if stream:
            def generate():
                for chunk in api.chat(body):
                    if chunk:
                        data = json.dumps({
                            "choices": [{"delta": {"content": chunk}}],
                            "object": "chat.completion.chunk"
                        })
                        yield f"data: {data}\n\n"
                yield "data: [DONE]\n\n"
            
            return Response(
                stream_with_context(generate()),
                mimetype="text/event-stream"
            )
        else:
            response_text = ""
            for chunk in api.chat(body):
                if chunk:
                    response_text += chunk
                    
            return jsonify({
                "choices": [{"message": {"content": response_text}}],
                "object": "chat.completion"
            })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/generate", methods=["POST"])
def generate():
    """Endpoint for non-streaming text generation"""
    try:
        api = get_api()
        body = request.json
        
        body["stream"] = False
        
        result = api.generate(body)
        return jsonify({
            "content": result,
            "object": "text.completion"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def create_app():
    """Create the Flask application"""
    return app

def run_server(host="0.0.0.0", port=13985):
    """Run the Flask server with debug WSGI"""
    app.run(host=host, port=port, debug=True)

if __name__ == "__main__":
    run_server()