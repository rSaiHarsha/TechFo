# Define get_embeddings at the module level

import os
import datetime
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from pymongo import MongoClient
import gridfs
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
import PyPDF2

from dotenv import load_dotenv

import requests

# Load environment variables from .env
load_dotenv()

def get_embeddings(texts):
    return get_mistral_embeddings(texts)

# Connection check for Qdrant and MongoDB
def check_connections():
    qdrant_ok = False
    mongo_ok = False
    try:
        # Qdrant: try to get collections list
        qdrant_client.get_collections()
        qdrant_ok = True
    except Exception:
        qdrant_ok = False
    try:
        # MongoDB: try to list collections
        db.list_collection_names()
        mongo_ok = True
    except Exception:
        mongo_ok = False
    return qdrant_ok, mongo_ok

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'txt'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = os.environ.get('FLASK_SECRET', 'supersecret')

# Env / clients
QDRANT_URL = os.environ.get('QDRANT_URL', 'http://localhost:6333')
QDRANT_API_KEY = os.environ.get('QDRANT_API_KEY')
MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017')

mongo = MongoClient(MONGO_URI)
db = mongo['tech_collections_db']

collections_meta = db['collections_meta']
# Setup GridFS
fs = gridfs.GridFS(db)

# Qdrant client (cloud cluster)
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)


# Mistral API endpoint
MISTRAL_API_URL = "https://api.mistral.ai/v1/embeddings"
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

def get_mistral_embeddings(texts):
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "mistral-embed",
        "input": texts,
    }
    response = requests.post(MISTRAL_API_URL, json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()
    # Return only the embeddings as a list of lists
    return [item["embedding"] for item in data["data"]]

# Helpers
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(path):
    text_chunks = []
    with open(path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for p in reader.pages:
            text = p.extract_text() or ''
            text_chunks.append(text)
    return "\n".join(text_chunks)


@app.route('/clear_collection/<collection_name>', methods=['POST'])
def clear_collection(collection_name):
    try:
        # Use FilterSelector with an empty Filter to delete all points
        qdrant_client.delete(
            collection_name=collection_name,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter()
            )
        )
        collections_meta.update_one({'name': collection_name}, {'$set': {'docs_count': 0}})
        flash(f'Cleared all data in collection {collection_name}', 'success')
    except Exception as e:
        flash(f'Failed to clear collection: {e}', 'danger')
    return redirect(url_for('view_collection', collection_name=collection_name))

@app.route('/', methods=['GET', 'POST'])
def index():
    cols = list(collections_meta.find({}))
    search_results = None
    search_query = None
    selected_collection = None
    # Check DB connections and alert if both are connected
    qdrant_ok, mongo_ok = check_connections()
    if qdrant_ok and mongo_ok:
        flash('Qdrant and MongoDB are connected.', 'success')
    elif not qdrant_ok:
        flash('Qdrant connection failed.', 'danger')
    elif not mongo_ok:
        flash('MongoDB connection failed.', 'danger')

    if request.method == 'POST':
        search_query = request.form.get('search_query')
        selected_collection = request.form.get('search_collection')
        if search_query and selected_collection:
            query_vec = get_embeddings([search_query])[0]
            try:
                if selected_collection == "__all__":
                    # Search all collections
                    search_results = []
                    for c in cols:
                        try:
                            result = qdrant_client.search(
                                collection_name=c['name'],
                                query_vector=query_vec,
                                limit=5
                            )
                            for r in result:
                                search_results.append({
                                    'score': r.score,
                                    'source': r.payload.get('source', 'N/A'),
                                    'collection': r.payload.get('collection', c['name']),
                                    'chunk': r.payload.get('chunk') or r.payload.get('page_content', 'N/A'),
                                })
                        except Exception as e:
                            continue
                    # Sort by score descending
                    search_results = sorted(search_results, key=lambda x: x['score'], reverse=True)[:10]
                else:
                    result = qdrant_client.search(
                        collection_name=selected_collection,
                        query_vector=query_vec,
                        limit=10
                    )
                    search_results = [
                        {
                            'score': r.score,
                            'source': r.payload.get('source', 'N/A'),
                            'collection': r.payload.get('collection', 'N/A'),
                            'chunk': r.payload.get('chunk') or r.payload.get('page_content', 'N/A'),
                        } for r in result
                    ]
            except Exception as e:
                flash(f'Search error: {e}', 'danger')
    return render_template('index.html', collections=cols, search_results=search_results, search_query=search_query, selected_collection=selected_collection)

@app.route('/collections')
def list_collections():
    cols = list(collections_meta.find({}))
    return render_template('collections.html', collections=cols)

@app.route('/create_collection', methods=['POST'])
def create_collection():
    name = request.form.get('name')
    description = request.form.get('description', '')
    if not name:
        flash('Collection name required', 'danger')
        return redirect(url_for('index'))


    existing = collections_meta.find_one({'name': name})
    if existing:
        flash('Collection already exists', 'warning')
        return redirect(url_for('index'))

    meta = {
        'name': name,
        'description': description,
        'created_at': datetime.datetime.utcnow(),
        'docs_count': 0
    }
    collections_meta.insert_one(meta)

    try:
        qdrant_client.recreate_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(size=1024, distance=qmodels.Distance.COSINE),
        )
    except Exception as e:
        print('qdrant create collection failed:', e)

    flash(f'Collection {name} created', 'success')
    return redirect(url_for('index'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    cols = list(collections_meta.find({}))
    if request.method == 'GET':
        return render_template('upload.html', collections=cols)

    file = request.files.get('file')
    collection_name = request.form.get('collection')
    if not file or file.filename == '':
        flash('No file selected', 'danger')
        return redirect(url_for('upload'))
    if not collection_name:
        flash('Select a collection', 'danger')
        return redirect(url_for('upload'))
    if not allowed_file(file.filename):
        flash('File type not allowed', 'danger')
        return redirect(url_for('upload'))


    filename = secure_filename(file.filename)

    # Save file in MongoDB GridFS
    file_id = fs.put(file, filename=filename, collection=collection_name)


    # Read file back from GridFS
    grid_out = fs.get(file_id)
    content = grid_out.read()

    # Extract text depending on file type
    if filename.lower().endswith('.pdf'):
        with open("temp.pdf", "wb") as temp:
            temp.write(content)
        text = extract_text_from_pdf("temp.pdf")
        os.remove("temp.pdf")
    else:
        text = content.decode('utf-8')

    if not text.strip():
        flash('No text extracted from file', 'warning')
        return redirect(url_for('upload'))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_text(text)
    documents = [Document(page_content=t, metadata={'source': filename}) for t in docs]

    vectors = get_embeddings([d.page_content for d in documents])

    points = []
    for i, vec in enumerate(vectors):
        payload = documents[i].metadata.copy()
        payload.update({'collection': collection_name, 'chunk': documents[i].page_content, 'page_content': documents[i].page_content})
        points.append(qmodels.PointStruct(id=i, vector=vec, payload=payload))

    qdrant_client.upsert(collection_name=collection_name, points=points)

    collections_meta.update_one({'name': collection_name}, {'$inc': {'docs_count': len(documents)}})

    flash(f'Uploaded and indexed {len(documents)} chunks to collection {collection_name}', 'success')
    return redirect(url_for('index'))

    query_vec = get_embeddings([search_query])[0]
def view_collection(collection_name):
    meta = collections_meta.find_one({'name': collection_name})
    try:
        resp = qdrant_client.scroll(collection_name=collection_name, limit=20)
        points = [{'id': p.id, 'payload': p.payload} for p in resp.points]
    except Exception as e:
        print('qdrant scroll error', e)
        points = []

    return render_template('view_collection.html', meta=meta, points=points)

if __name__ == '__main__':
    app.run(debug=False)
