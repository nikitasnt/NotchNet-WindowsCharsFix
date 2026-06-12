import json
import logging
import multiprocessing
import os
import threading
import time
import traceback

from flask import Flask, request, jsonify, Response, stream_with_context  # type: ignore
from flask_cors import CORS  # type: ignore
from flask_limiter import Limiter  # type: ignore
from flask_limiter.util import get_remote_address  # type: ignore

from config.rag_pipeline import generate_answer, reload_qa_chain
from config import config
from config import build_index
from wiki import wiki_loader
from wiki import clean_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)
CORS(app)

# Global state for indexing
indexing_lock = threading.Lock()
in_progress_wikis = set()
in_progress_lock = threading.Lock()
PROCESSED_WIKIS_FILE = os.path.join(config.DATA_DIR_CLEANED, "processed_wikis.json")


def load_processed_wikis():
    if os.path.exists(PROCESSED_WIKIS_FILE):
        try:
            with open(PROCESSED_WIKIS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.warning("Failed to load processed wikis: %s", e)
            return {}
    return {}


def save_processed_wiki(wiki_url):
    data = load_processed_wikis()
    data[wiki_url] = time.time()
    os.makedirs(os.path.dirname(PROCESSED_WIKIS_FILE), exist_ok=True)
    with open(PROCESSED_WIKIS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _require_api_key():
    if config.LOCAL_MODE:
        return
    api_key = request.headers.get("X-API-Key", "")
    if api_key != config.API_KEY:
        logger.warning("Unauthorized request from %s", get_remote_address())
        return jsonify({"error": "Unauthorized. Invalid or missing X-API-Key header."}), 401
    return None


@app.route("/ask", methods=["POST"])
@limiter.limit("5 per minute")
def ask_question():
    auth = _require_api_key()
    if auth:
        return auth

    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' field"}), 400

    try:
        answer = generate_answer(data["question"])
        return jsonify({"answer": answer})
    except Exception as e:
        logger.exception("Error generating answer")
        return jsonify({"error": "Server error", "details": str(e)}), 500


@app.route("/ask/stream", methods=["POST"])
@limiter.limit("5 per minute")
def ask_question_stream():
    auth = _require_api_key()
    if auth:
        return auth

    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' field"}), 400

    q_len = len(data["question"])
    logger.info("Received question of length: %d characters", q_len)
    if q_len > 10000:
        logger.warning("Massive input detected! First 500 chars: %s", data["question"][:500])

    from config.rag_pipeline import generate_answer_stream

    def generate():
        try:
            for token in generate_answer_stream(data["question"]):
                payload = json.dumps({"answer": token})
                yield f"data: {payload}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.exception("Error in stream generation")
            yield f'data: {{"error": "{str(e)}"}}\n\n'

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


def background_wiki_processing(api_url, categories, force=False):
    with in_progress_lock:
        if api_url in in_progress_wikis:
            logger.info("Wiki %s is already being processed. Skipping.", api_url)
            return

    # Check cache if not forced
    if not force:
        processed = load_processed_wikis()
        last_time = processed.get(api_url, 0)
        if time.time() - last_time < 86400:
            logger.info("Wiki %s was processed recently. Skipping download.", api_url)
            return

    with in_progress_lock:
        in_progress_wikis.add(api_url)

    logger.info("Starting background wiki processing for %s...", api_url)
    try:
        wiki_loader.fetch_wiki(api_url, set(categories))
        clean_data.walk_and_clean()

        with indexing_lock:
            build_index.build_index()
            reload_qa_chain()

        save_processed_wiki(api_url)
        logger.info("Background wiki processing complete for %s!", api_url)
    except Exception as e:
        logger.error("Background wiki processing failed for %s: %s", api_url, e)
        logger.exception("Traceback")
    finally:
        with in_progress_lock:
            in_progress_wikis.discard(api_url)


@app.route("/admin/add-wiki", methods=["POST"])
@limiter.limit("3 per minute")
def add_wiki():
    auth = _require_api_key()
    if auth:
        return auth

    data = request.get_json()
    if not data or "categories" not in data:
        return jsonify({"error": "Missing 'categories' list"}), 400

    api_url = data.get("api_url", config.WIKI_API_URL_DEFAULT)
    categories = data["categories"]

    thread = threading.Thread(target=background_wiki_processing, args=(api_url, categories, True))
    thread.start()

    return jsonify({"status": "processing_started", "message": "Wiki download and indexing started in background."})


@app.route("/admin/detect-mods", methods=["POST"])
@limiter.limit("3 per minute")
def detect_mods():
    return jsonify({
        "status": "success",
        "processed_mods": 0,
        "details": [],
        "message": "Mod detection and wiki search is currently disabled.",
    })


@app.route("/admin/reload-index", methods=["POST"])
@limiter.limit("3 per minute")
def reload_index():
    try:
        reload_qa_chain()
        return jsonify({"status": "success", "message": "QA Chain reloaded."})
    except Exception as e:
        logger.exception("Error reloading QA chain")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    app.run(host="0.0.0.0", port=8000, debug=False)
