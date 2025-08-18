import os
import json
import telebot
import logging
import uuid
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton

# --- Load Configuration ---
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
QDRANT_PATH = os.getenv("QDRANT_PATH")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")

# --- Constants ---
LOG_FILE = 'jokes_log.json'

# --- Validate Configuration ---
for var in [BOT_TOKEN, OPENAI_API_KEY, OPENAI_BASE_URL, QDRANT_PATH, QDRANT_COLLECTION_NAME]:
    if not var:
        raise ValueError(f"Environment variable for {var} is not set. Please check your .env file.")

# --- Initialize Services ---
bot = telebot.TeleBot(BOT_TOKEN)
logger = telebot.logger
telebot.logger.setLevel(logging.INFO)

openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
qdrant_client = QdrantClient(path=QDRANT_PATH)
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# In-memory storage for sent jokes to track votes
jokes_sent = {}

# --- JSON Logging Functions ---
def append_log_entry(entry):
    """Appends a new log entry to the JSON log file."""
    try:
        with open(LOG_FILE, 'r+', encoding='utf-8') as f:
            logs = json.load(f)
            logs.append(entry)
            f.seek(0)
            json.dump(logs, f, ensure_ascii=False, indent=4)
    except (FileNotFoundError, json.JSONDecodeError):
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump([entry], f, ensure_ascii=False, indent=4)

def update_log_rating(joke_id, rating):
    """Updates the rating for a specific joke in the JSON log file."""
    try:
        with open(LOG_FILE, 'r+', encoding='utf-8') as f:
            logs = json.load(f)
            for entry in logs:
                if entry.get('id') == joke_id:
                    entry['rating'] = rating
                    break
            f.seek(0)
            f.truncate()
            json.dump(logs, f, ensure_ascii=False, indent=4)
    except (FileNotFoundError, json.JSONDecodeError):
        logger.error(f"Could not update rating. Log file '{LOG_FILE}' not found or is corrupted.")

# --- Qdrant Setup Function ---
def setup_qdrant():
    """Initializes the Qdrant collection, populating it if it doesn't exist."""
    try:
        qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        logger.info(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' already exists.")
    except Exception:
        logger.info(f"Creating Qdrant collection '{QDRANT_COLLECTION_NAME}'.")

        vector_size = embedding_model.get_sentence_embedding_dimension()
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
        )

        try:
            with open('jokes.json', 'r', encoding='utf-8') as f:
                jokes = json.load(f)
        except FileNotFoundError:
            logger.error("jokes.json not found. Please create the file.")
            return

        documents = list(jokes.values())
        
        logger.info("Generating embeddings for documents...")
        vectors = embedding_model.encode(documents, show_progress_bar=True)

        qdrant_client.upload_points(
            collection_name=QDRANT_COLLECTION_NAME,
            points=[
                models.PointStruct(id=idx, vector=vector.tolist(), payload={"text": doc})
                for idx, (vector, doc) in enumerate(zip(vectors, documents))
            ],
            wait=True
        )
        logger.info("Successfully populated Qdrant collection.")

# --- RAG Helper Function ---
def get_rag_response(prompt: str):
    """Retrieves context, generates a response, and returns both."""
    try:
        query_vector = embedding_model.encode(prompt).tolist()

        search_result = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=query_vector,
            limit=6
        )

        relevant_jokes_list = [hit.payload.get("text", "") for hit in search_result] if search_result else []
        context_jokes = '\n\n__________'.join(relevant_jokes_list)

        augmented_prompt = (
            f"Тебе будет дано шесть анекдотов и тема от пользователя. Тебе надо выбрать самый подходящий под тему анекдот и переделать его под тему от пользователя. Не добавляй никаких объяснений или форматирования, отвечай только анекдотом.\n\n"
            "ПРИМЕР ПЕРЕДЕЛЫВАНИЯ 1:\n\n"
            "ТЕМА АНЕКДОТА: Отношения Степана и его жены\n\n"
            """АНЕКДОТ ДЛЯ ПЕРЕДЕЛКИ:Мужику звонят из милиции и говорят:
— У нас для вас три новости — плохая, хорошая и охуенная.
— Гм, ну давайте с плохой.
— Ваша жена погибла — утонула в реке.
— А хорошая?
— Когда мы ее достали — ее тело было облеплено раками и мы заебато попили пиво всем отделом.
— Гм, а охуенная тогда какая?— Мы ее снова забросили в реку и приглашаем вас завтра на пиво!)
\n\n"""
            """ПЕРЕДЕЛАННЫЙ АНЕКДОТ:Степану звонят из милиции и говорят:
— Степан, здравствуйте, у нас для вас три новости — плохая, хорошая и охуенная.
— Гм, ну давайте с плохой.
— Ваша жена погибла — утонула в реке.
— А хорошая?
— Когда мы ее достали — ее тело было облеплено раками и мы заебато попили пиво всем отделом.
— Гм, а охуенная тогда какая?— Мы ее снова забросили в реку и приглашаем вас завтра на пиво!)
\n\n"""
            f"АНЕКДОТЫ, ОДИН ИЗ КОТОРЫХ НАДО ПЕРЕДЕЛАТЬ:\n\"{context_jokes}\"\n\n"
            f"ТЕМА ОТ ПОЛЬЗОВАТЕЛЯ:\n\"{prompt}\""
        )

        response = openai_client.chat.completions.create(
            model="Qwen/Qwen3-235B-A22B-Instruct-2507",
            max_tokens=1024,
            temperature=0.8,
            messages=[
                {"role": "system", "content": "Ты — креативный ИИ-юморист. Твоя задача — проанализировать теорию юмора, пример и тему от пользователя, а затем сгенерировать на этой основе остроумную шутку."},
                {"role": "user", "content": augmented_prompt}
            ]
        )
        generated_joke = response.choices[0].message.content
        return generated_joke, relevant_jokes_list
    except Exception as e:
        logger.error(f"RAG process failed: {e}")
        return "Извините, у меня творческий кризис. Попробуйте еще раз.", []

# --- Bot Handlers ---
@bot.message_handler(commands=['start'])
def send_welcome(message):
    """Sends a welcome message when the /start command is issued."""
    bot.reply_to(message, "Привет! Давай-ка пошуткуем. Кидай тему, я попробую придумать анекдот.")

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    """Handles messages, gets a joke, logs it, and sends it with vote buttons."""
    if message.text.startswith('/'):
        return

    response_text, relevant_jokes = get_rag_response(message.text)
    
    joke_id = str(uuid.uuid4())
    jokes_sent[joke_id] = response_text
    
    log_entry = {
        'id': joke_id,
        'prompt': message.text,
        'generated_joke': response_text,
        'rating': 0,
        'relevant_jokes': relevant_jokes
    }
    append_log_entry(log_entry)
    
    markup = InlineKeyboardMarkup()
    upvote_button = InlineKeyboardButton("👍", callback_data=f"vote:up:{joke_id}")
    downvote_button = InlineKeyboardButton("👎", callback_data=f"vote:down:{joke_id}")
    markup.add(upvote_button, downvote_button)
    
    bot.reply_to(message, response_text, reply_markup=markup)

@bot.callback_query_handler(func=lambda call: call.data.startswith('vote:'))
def handle_vote(call):
    """Handles button presses, updates the log file, and removes buttons."""
    try:
        _, action, joke_id = call.data.split(':')
        rating = 1 if action == 'up' else -1
        
        update_log_rating(joke_id, rating)
        
        joke_text = jokes_sent.get(joke_id, "Неизвестный анекдот")
        logger.info(f"User '{call.from_user.username}' voted '{action}' for joke: '{joke_text[:70]}...'")
        
        bot.answer_callback_query(call.id, text="Спасибо за оценку!")
        bot.edit_message_reply_markup(chat_id=call.message.chat.id, message_id=call.message.message_id, reply_markup=None)
    except Exception as e:
        logger.error(f"Error handling vote: {e}")
        bot.answer_callback_query(call.id, text="Ошибка при обработке голоса.")

def main():
    """Set up Qdrant and start the bot."""
    setup_qdrant()
    print("Bot is running... Press Ctrl+C to stop.")
    bot.infinity_polling()

if __name__ == '__main__':
    main()

