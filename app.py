import os
import re
import json
import logging
import aiohttp
import asyncio
import nltk
from typing import List, Dict, Optional
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    CallbackQueryHandler,
    filters,
    MessageHandler
)
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from dateutil import parser as date_parser
import os
import shutil
import nltk
from dotenv import load_dotenv

nltk_data_path = "/app/nltk_data"

# Überprüfen, ob der Pfad eine Datei oder ein Verzeichnis ist
if os.path.exists(nltk_data_path):
    try:
        if os.path.isdir(nltk_data_path):
            shutil.rmtree(nltk_data_path)  # Falls es ein Verzeichnis ist, lösche es
        elif os.path.isfile(nltk_data_path):
            os.remove(nltk_data_path)  # Falls es eine Datei ist, lösche sie
    except PermissionError as e:
        print(f"Zugriffsfehler: {e}. Kann das Verzeichnis oder die Datei nicht löschen.")
else:
    print(f"{nltk_data_path} existiert nicht.")

# Verzeichnis erneut erstellen
os.makedirs(nltk_data_path, exist_ok=True)

# Setze Umgebungsvariable für NLTK
os.environ["NLTK_DATA"] = nltk_data_path

# Stoppwörter herunterladen
nltk.download("stopwords", download_dir=nltk_data_path)

# Testen, ob das Verzeichnis korrekt gesetzt wurde
print(nltk.data.path)

# Lade Umgebungsvariablen (falls benötigt)
load_dotenv()
# Logging Konfiguration mit sicherer Pfadangabe
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("/tmp/newsbot.log"),  # Temporäres Verzeichnis mit Schreibrechten
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Konfiguration
CONFIG = {
    'NEWS_API_KEY': '4ba3cb8c668e466398a33a8f4a35e76a',  # Korrigierte Umgebungsvariablen-Namen
    'TELEGRAM_TOKEN': '7562186384:AAEvykmSQHkdkFMQhq8FJss',
    'CACHE_TTL': 300,
    'MAX_ARTICLES': 15,
    'SUMMARY_SENTENCES': 3,
    'SOURCE_WEIGHTS': {
        'spiegel-online': 1.0,
        'zeit': 0.9,
        'tagesschau': 0.95,
        'reuters': 0.85,
        'the-washington-post': 0.8
    },
    'LANGUAGES': {
        'de': {'sumy': 'german', 'stopwords': 'german'},
        'en': {'sumy': 'english', 'stopwords': 'english'}
    }
}

class NewsCache:
    def __init__(self):
        self.cache = {}
        self.timestamps = {}

    def add(self, key: str, data: List[Dict]):
        self.cache[key] = data
        self.timestamps[key] = datetime.datetime.now()

    def get(self, key: str) -> Optional[List[Dict]]:
        if key in self.cache:
            age = (datetime.datetime.now() - self.timestamps[key]).seconds
            if age < CONFIG['CACHE_TTL']:
                return self.cache[key]
        return None

class AdvancedNewsFetcher:
    def __init__(self):
        self.base_url = "https://newsapi.org/v2/"
        self.headers = {"X-Api-Key": CONFIG['NEWS_API_KEY']}
        self.cache = NewsCache()
        self.session = aiohttp.ClientSession()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def fetch_news(self, params: Dict) -> List[Dict]:
        cache_key = json.dumps(params, sort_keys=True)
        if cached := self.cache.get(cache_key):
            return cached
        
        try:
            async with self.session.get(
                f"{self.base_url}everything",
                headers=self.headers,
                params=params,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as response:
                response.raise_for_status()
                data = await response.json()
                processed = self._process_articles(data['articles'])
                self.cache.add(cache_key, processed)
                return processed
        except Exception as e:
            logger.error(f"News API Error: {str(e)}")
            return []
        finally:
            await self.session.close()

    def _process_articles(self, articles: List[Dict]) -> List[Dict]:
        processed = []
        for article in articles:
            if self._is_valid(article):
                article['score'] = self._calculate_article_score(article)
                processed.append(article)
        return self._sort_articles(processed)[:CONFIG['MAX_ARTICLES']]

    def _is_valid(self, article: Dict) -> bool:
        required = {
            'title': (str, 10),
            'content': (str, 300),
            'url': (str, 10),
            'publishedAt': (str, 10)
        }
        return all(isinstance(article.get(k), t) and len(str(article.get(k))) > l 
               for k, (t, l) in required.items())

    def _calculate_article_score(self, article: Dict) -> float:
        source_score = CONFIG['SOURCE_WEIGHTS'].get(article['source']['id'], 0.7)
        content_length = len(article['content'])
        time_score = 1 - (datetime.datetime.now() - 
                        date_parser.parse(article['publishedAt'])).total_seconds() / 86400
        return (source_score * 0.5 + 
               min(content_length/2000, 1) * 0.3 + 
               time_score * 0.2)

    def _sort_articles(self, articles: List[Dict]) -> List[Dict]:
        return sorted(
            articles,
            key=lambda x: (-x['score'], x['publishedAt']),
            reverse=False  # Korrigierte Sortierreihenfolge
        )

class AdvancedSummarizer:
    def __init__(self):
        self.summarizers = {
            'lsa': LsaSummarizer(),
            'lexrank': LexRankSummarizer()
        }

    def summarize(self, text: str, language: str = 'en') -> str:
        try:
            parser = PlaintextParser.from_string(text, Tokenizer(CONFIG['LANGUAGES'][language]['sumy']))
            summarizer = self.summarizers['lexrank']
            summarizer.stop_words = self._get_stop_words(language)
            
            sentences = summarizer(parser.document, CONFIG['SUMMARY_SENTENCES'])
            return ' '.join([str(s) for s in sentences])
        except Exception as e:
            logger.error(f"Summarization error: {str(e)}")
            return text[:500] + " [...]"

    def _get_stop_words(self, language: str):
        return set(nltk.corpus.stopwords.words(CONFIG['LANGUAGES'][language]['stopwords']))

class NewsBot:
    def __init__(self):
        self.fetcher = AdvancedNewsFetcher()
        self.summarizer = AdvancedSummarizer()
        self.user_prefs = {}

    async def handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        keyboard = [
            [InlineKeyboardButton("🇩🇪 Deutsche News", callback_data='de')],
            [InlineKeyboardButton("🌍 International News", callback_data='en')],
            [InlineKeyboardButton("⚙️ Einstellungen", callback_data='settings')]
        ]
        await update.message.reply_text(
            "📰 *News Professional Bot*\nWählen Sie eine Kategorie:",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='Markdown'
        )

    async def send_news(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        language = query.data
        
        params = self._build_query_params(language)
        articles = await self.fetcher.fetch_news(params)
        
        if not articles:
            await query.edit_message_text("⚠️ Keine aktuellen Nachrichten gefunden.")
            return

        for article in articles:
            await self._send_article(query, article, language)

    def _build_query_params(self, language: str) -> Dict:
        return {
            'language': language,
            'sortBy': 'publishedAt',
            'pageSize': CONFIG['MAX_ARTICLES'],
            'q': 'news NOT sport NOT entertainment',
            'excludeDomains': 'twitter.com,facebook.com'
        }

    async def _send_article(self, query, article: Dict, language: str):
        summary = self.summarizer.summarize(article['content'], language)
        message = self._format_message(article, summary)
        
        markup = InlineKeyboardMarkup([[
            InlineKeyboardButton("🔗 Artikel lesen", url=article['url'])
        ]])
        
        try:
            if article.get('urlToImage'):
                await query.message.reply_photo(
                    photo=article['urlToImage'],
                    caption=message,
                    parse_mode='Markdown',
                    reply_markup=markup
                )
            else:
                await query.message.reply_text(
                    message,
                    parse_mode='Markdown',
                    reply_markup=markup
                )
        except Exception as e:
            logger.error(f"Error sending article: {str(e)}")

    def _format_message(self, article: Dict, summary: str) -> str:
        return (
            f"🔥 *{article['title']}*\n\n"
            f"{summary}\n\n"
            f"🏅 Bewertung: {article['score']:.2f}/1.0\n"
            f"📡 Quelle: {article['source']['name']}\n"
            f"🕒 {date_parser.parse(article['publishedAt']).strftime('%d.%m.%Y %H:%M')}"
        )

def main():
    app = Application.builder().token(CONFIG['TELEGRAM_TOKEN']).build()
    news_bot = NewsBot()

    # Handler registrieren
    app.add_handler(CommandHandler("start", news_bot.handle_start))
    app.add_handler(CallbackQueryHandler(news_bot.send_news, pattern="^(de|en)$"))
    
    # Fehlerbehandlung
    app.add_error_handler(lambda update, context: logger.error(context.error))

    app.run_polling()

if __name__ == "__main__":
    main()