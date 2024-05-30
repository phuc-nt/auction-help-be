from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import openai
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import markdown

# Load environment variables from .env file
load_dotenv()

# Get Qdrant API key from environment
qdrant_api_key = os.getenv('QDRANT_API_KEY')

# Initialize Qdrant client
client_qdrant = QdrantClient(
    url="https://5d9b085c-df8b-4f83-81f2-82d006da134a.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key=qdrant_api_key
)
collection_name = 'auction_help_1_text-embedding-3-large'

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Phúc Đẹp Trai"

def get_embedding(text, model):
    text = str(text).replace("\n", " ")  # Loại bỏ xuống dòng
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding, response.usage.total_tokens

def search_similar_sentences(query_embedding):
    return client_qdrant.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=3  # Lấy 3 kết quả hàng đầu
    )

def format_response(query, results):
    response = f"Truy vấn: {query}\n"
    for i, result in enumerate(results, start=1):
        snippet = result.payload['text']
        url = result.payload['url']
        title = result.payload['title']
        response += f"\nKết quả {i}: {snippet}\nURL: {url}\nTitle: {title}\n"
    return response

def ask_gpt(query, context):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Bạn là trợ lý đa ngôn ngữ về dịch vụ Yahoo!オークション, bạn chuyên trả lời các câu hỏi về cách sử dụng dịch vụ, dựa trên nội dung câu hỏi và ngữ cảnh. Bạn sẽ đưa ra câu trả lời tổng hợp ngắn gọn, và dẫn các link của bài viết chi tiết để người dùng tìm hiểu thêm. Ngữ cảnh: " + context},
            {"role": "user", "content": query}
        ]
    )
    return completion.choices[0].message.content

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    messages = data['messages']

    # Prepare messages for OpenAI API
    user_message = messages[-1]['content']
    openai_messages = [{"role": msg['role'], "content": msg['content']} for msg in messages]

    # Embed the last message using the get_embedding function
    last_message_embedding, _ = get_embedding(user_message, "text-embedding-3-large")

    # Search in Qdrant
    search_result = search_similar_sentences(last_message_embedding)

    # Format the search result for context
    formatted_context = format_response(user_message, search_result)

    # Use OpenAI to generate a response
    bot_response = ask_gpt(user_message, formatted_context)
    
    # Convert Markdown to HTML
    html_response = markdown.markdown(bot_response)
    
    # print(html_response)

    return html_response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
