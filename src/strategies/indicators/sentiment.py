"""Sentiment analysis with DeepSeek LLM integration"""

import openai
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
import logging
import time
import re
from datetime import datetime, timedelta
from dataclasses import dataclass
import requests
import jieba
from collections import Counter
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

from config import DEEPSEEK_API_KEY

logger = logging.getLogger(__name__)

@dataclass
class SentimentData:
    score: float  # -1 to 1
    confidence: float  # 0 to 1
    keywords: List[str]
    source: str
    timestamp: datetime

@dataclass
class NewsItem:
    title: str
    content: str
    source: str
    timestamp: datetime
    symbol: Optional[str] = None

class DeepSeekSentimentAnalyzer:
    """DeepSeek LLM integration for news sentiment analysis"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or DEEPSEEK_API_KEY
        if not self.api_key:
            raise ValueError("DeepSeek API key is required")
        
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com/v1"
        )
        
        self.rate_limit_delay = 1.0  # seconds between requests
        self.max_retries = 3
        
    def analyze_single_text(self, text: str, symbol: str = None) -> SentimentData:
        """Analyze sentiment of a single text"""
        
        prompt = self._create_sentiment_prompt(text, symbol)
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a professional financial analyst specialized in A-Share market sentiment analysis. Provide precise numerical sentiment scores."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.1,
                    max_tokens=200
                )
                
                result = self._parse_sentiment_response(response.choices[0].message.content)
                keywords = self._extract_keywords(text)
                
                return SentimentData(
                    score=result['sentiment'],
                    confidence=result['confidence'],
                    keywords=keywords,
                    source='deepseek',
                    timestamp=datetime.now()
                )
                
            except Exception as e:
                logger.error(f"DeepSeek API error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.rate_limit_delay * (2 ** attempt))
                else:
                    # Fallback to rule-based analysis
                    return self._fallback_sentiment_analysis(text)
            
            time.sleep(self.rate_limit_delay)
    
    def analyze_batch_texts(self, texts: List[str], symbols: List[str] = None) -> List[SentimentData]:
        """Analyze sentiment for multiple texts"""
        if symbols is None:
            symbols = [None] * len(texts)
        
        results = []
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(self.analyze_single_text, text, symbol)
                for text, symbol in zip(texts, symbols)
            ]
            
            for future in futures:
                try:
                    result = future.result(timeout=30)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch analysis error: {e}")
                    results.append(SentimentData(0.0, 0.0, [], 'error', datetime.now()))
        
        return results
    
    def _create_sentiment_prompt(self, text: str, symbol: str = None) -> str:
        """Create prompt for sentiment analysis"""
        symbol_context = f" for stock {symbol}" if symbol else ""
        
        return f"""
        请分析以下中文财经新闻{symbol_context}的市场情绪影响：

        新闻内容：{text}

        请提供以下信息：
        1. 情绪评分：-1到1之间的数值（-1极度负面，0中性，1极度正面）
        2. 置信度：0到1之间的数值（对评分的确信程度）
        3. 关键影响因素：简要说明

        格式：
        情绪评分：[数值]
        置信度：[数值]
        影响因素：[简要说明]
        """
    
    def _parse_sentiment_response(self, response_text: str) -> Dict[str, float]:
        """Parse sentiment response from DeepSeek"""
        try:
            # Extract sentiment score
            sentiment_match = re.search(r'情绪评分[：:]?\s*(-?\d*\.?\d+)', response_text)
            sentiment = float(sentiment_match.group(1)) if sentiment_match else 0.0
            
            # Extract confidence score  
            confidence_match = re.search(r'置信度[：:]?\s*(\d*\.?\d+)', response_text)
            confidence = float(confidence_match.group(1)) if confidence_match else 0.5
            
            # Ensure values are in valid ranges
            sentiment = max(-1.0, min(1.0, sentiment))
            confidence = max(0.0, min(1.0, confidence))
            
            return {
                'sentiment': sentiment,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error parsing sentiment response: {e}")
            return {'sentiment': 0.0, 'confidence': 0.0}
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from Chinese text"""
        try:
            # Financial keywords
            financial_keywords = [
                '业绩', '利润', '收入', '营收', '盈利', '亏损', '增长', '下降',
                '投资', '并购', '重组', '上市', '退市', '停牌', '复牌',
                '政策', '监管', '调控', '刺激', '支持', '限制',
                '市场', '行业', '竞争', '份额', '领先', '落后',
                '创新', '技术', '研发', '专利', '产品', '服务'
            ]
            
            # Segment text
            words = jieba.lcut(text)
            
            # Filter financial keywords
            keywords = [word for word in words if word in financial_keywords and len(word) > 1]
            
            # Get top keywords by frequency
            keyword_freq = Counter(keywords)
            top_keywords = [word for word, _ in keyword_freq.most_common(5)]
            
            return top_keywords
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def _fallback_sentiment_analysis(self, text: str) -> SentimentData:
        """Fallback rule-based sentiment analysis"""
        # Positive keywords
        positive_words = [
            '增长', '上涨', '盈利', '收益', '利好', '强劲', '优秀', '超预期',
            '创新', '突破', '领先', '成功', '合作', '扩张', '投资'
        ]
        
        # Negative keywords
        negative_words = [
            '下跌', '亏损', '下降', '风险', '担忧', '困难', '挑战', '压力',
            '减少', '削减', '裁员', '关闭', '退出', '失败', '损失'
        ]
        
        # Count sentiment words
        words = jieba.lcut(text)
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        # Calculate sentiment score
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            sentiment = 0.0
            confidence = 0.1
        else:
            sentiment = (positive_count - negative_count) / max(len(words), 10)
            confidence = min(total_sentiment_words / 10, 0.8)
        
        return SentimentData(
            score=max(-1.0, min(1.0, sentiment)),
            confidence=confidence,
            keywords=self._extract_keywords(text),
            source='fallback',
            timestamp=datetime.now()
        )

class NewsDataProvider:
    """News data provider for A-Share market"""
    
    def __init__(self):
        self.sources = {
            'sina': 'https://finance.sina.com.cn',
            'eastmoney': 'https://finance.eastmoney.com',
            'cnstock': 'https://www.cnstock.com'
        }
    
    def fetch_latest_news(self, symbol: str = None, limit: int = 20) -> List[NewsItem]:
        """Fetch latest financial news"""
        # Placeholder implementation
        # In production, this would integrate with real news APIs
        
        sample_news = [
            {
                'title': f'A股市场今日表现强劲，{symbol or "多只股票"}收涨',
                'content': '今日A股市场整体表现良好，投资者情绪乐观，多个板块出现上涨态势。',
                'source': 'sina',
                'timestamp': datetime.now() - timedelta(hours=1)
            },
            {
                'title': f'{symbol or "相关公司"}发布业绩预告，预期增长显著',
                'content': '公司发布业绩预告，预计本季度净利润同比增长30%以上，超出市场预期。',
                'source': 'eastmoney', 
                'timestamp': datetime.now() - timedelta(hours=2)
            }
        ]
        
        return [
            NewsItem(
                title=news['title'],
                content=news['content'],
                source=news['source'],
                timestamp=news['timestamp'],
                symbol=symbol
            ) for news in sample_news[:limit]
        ]
    
    def search_news_by_keywords(self, keywords: List[str], days_back: int = 7) -> List[NewsItem]:
        """Search news by keywords"""
        # Placeholder implementation
        return []

class SentimentAggregator:
    """Aggregate sentiment from multiple sources"""
    
    def __init__(self, analyzer: DeepSeekSentimentAnalyzer):
        self.analyzer = analyzer
        self.news_provider = NewsDataProvider()
        self.sentiment_cache: Dict[str, List[SentimentData]] = {}
    
    def get_stock_sentiment(self, symbol: str, lookback_hours: int = 24) -> Dict[str, Any]:
        """Get aggregated sentiment for a stock"""
        
        # Fetch recent news
        news_items = self.news_provider.fetch_latest_news(symbol, limit=10)
        
        if not news_items:
            return self._default_sentiment()
        
        # Analyze sentiment
        texts = [f"{item.title} {item.content}" for item in news_items]
        sentiments = self.analyzer.analyze_batch_texts(texts, [symbol] * len(texts))
        
        # Aggregate results
        return self._aggregate_sentiments(sentiments, symbol)
    
    def get_market_sentiment(self, lookback_hours: int = 24) -> Dict[str, Any]:
        """Get overall market sentiment"""
        
        # Fetch general market news
        news_items = self.news_provider.fetch_latest_news(limit=20)
        
        if not news_items:
            return self._default_sentiment()
        
        # Analyze sentiment
        texts = [f"{item.title} {item.content}" for item in news_items]
        sentiments = self.analyzer.analyze_batch_texts(texts)
        
        # Aggregate results
        return self._aggregate_sentiments(sentiments, 'market')
    
    def _aggregate_sentiments(self, sentiments: List[SentimentData], identifier: str) -> Dict[str, Any]:
        """Aggregate multiple sentiment scores"""
        if not sentiments:
            return self._default_sentiment()
        
        # Calculate weighted average (weight by confidence)
        total_weight = sum(s.confidence for s in sentiments)
        
        if total_weight == 0:
            weighted_sentiment = 0.0
            avg_confidence = 0.0
        else:
            weighted_sentiment = sum(s.score * s.confidence for s in sentiments) / total_weight
            avg_confidence = total_weight / len(sentiments)
        
        # Collect all keywords
        all_keywords = []
        for s in sentiments:
            all_keywords.extend(s.keywords)
        
        keyword_freq = Counter(all_keywords)
        top_keywords = [word for word, _ in keyword_freq.most_common(5)]
        
        # Calculate sentiment distribution
        positive_count = sum(1 for s in sentiments if s.score > 0.1)
        negative_count = sum(1 for s in sentiments if s.score < -0.1)
        neutral_count = len(sentiments) - positive_count - negative_count
        
        # Cache result
        self.sentiment_cache[identifier] = sentiments
        
        return {
            'sentiment_score': round(weighted_sentiment, 3),
            'confidence': round(avg_confidence, 3),
            'samples_count': len(sentiments),
            'positive_ratio': positive_count / len(sentiments),
            'negative_ratio': negative_count / len(sentiments),
            'neutral_ratio': neutral_count / len(sentiments),
            'top_keywords': top_keywords,
            'timestamp': datetime.now(),
            'sources': list(set(s.source for s in sentiments))
        }
    
    def _default_sentiment(self) -> Dict[str, Any]:
        """Default sentiment when no data available"""
        return {
            'sentiment_score': 0.0,
            'confidence': 0.0,
            'samples_count': 0,
            'positive_ratio': 0.0,
            'negative_ratio': 0.0,
            'neutral_ratio': 1.0,
            'top_keywords': [],
            'timestamp': datetime.now(),
            'sources': []
        }
    
    def get_sentiment_trend(self, symbol: str, days: int = 7) -> pd.DataFrame:
        """Get sentiment trend over time"""
        # Placeholder implementation
        # In production, this would fetch historical sentiment data
        
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Generate sample trend data
        sentiment_scores = np.random.normal(0.0, 0.3, days)
        confidence_scores = np.random.uniform(0.3, 0.8, days)
        
        return pd.DataFrame({
            'date': dates,
            'sentiment_score': sentiment_scores,
            'confidence': confidence_scores,
            'positive_ratio': np.random.uniform(0.2, 0.6, days),
            'negative_ratio': np.random.uniform(0.1, 0.4, days)
        })

# Global sentiment analyzer instance
try:
    sentiment_analyzer = DeepSeekSentimentAnalyzer()
    sentiment_aggregator = SentimentAggregator(sentiment_analyzer)
except Exception as e:
    logger.warning(f"Could not initialize sentiment analyzer: {e}")
    sentiment_analyzer = None
    sentiment_aggregator = None

def get_stock_sentiment_score(symbol: str) -> float:
    """Simple function to get sentiment score for a stock"""
    if sentiment_aggregator is None:
        return 0.0
    
    try:
        sentiment_data = sentiment_aggregator.get_stock_sentiment(symbol)
        return sentiment_data['sentiment_score']
    except Exception as e:
        logger.error(f"Error getting sentiment for {symbol}: {e}")
        return 0.0