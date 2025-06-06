# API Documentation

## Overview
This document provides detailed information about the API endpoints, request/response formats, and usage examples.

## Base URL
```
http://localhost:8000/api/v1
```

## Authentication
All API requests require an API key to be included in the header:
```
X-API-Key: your_api_key_here
```

## Endpoints

### Market Data

#### Get Stock Price
```http
GET /market/price/{symbol}
```

**Parameters:**
- `symbol` (path): Stock symbol (e.g., AAPL, GOOGL)

**Response:**
```json
{
    "symbol": "AAPL",
    "price": 150.25,
    "timestamp": "2024-02-20T10:30:00Z",
    "volume": 1000000
}
```

#### Get Historical Data
```http
GET /market/historical/{symbol}
```

**Parameters:**
- `symbol` (path): Stock symbol
- `start_date` (query): Start date (YYYY-MM-DD)
- `end_date` (query): End date (YYYY-MM-DD)
- `interval` (query): Data interval (1m, 5m, 15m, 1h, 1d)

**Response:**
```json
{
    "symbol": "AAPL",
    "data": [
        {
            "timestamp": "2024-02-20T10:30:00Z",
            "open": 150.00,
            "high": 151.00,
            "low": 149.50,
            "close": 150.25,
            "volume": 1000000
        }
    ]
}
```

### Trading

#### Get Portfolio
```http
GET /trading/portfolio
```

**Response:**
```json
{
    "total_value": 100000.00,
    "positions": [
        {
            "symbol": "AAPL",
            "quantity": 100,
            "average_price": 150.00,
            "current_price": 150.25,
            "market_value": 15025.00,
            "unrealized_pnl": 25.00
        }
    ]
}
```

#### Place Order
```http
POST /trading/order
```

**Request Body:**
```json
{
    "symbol": "AAPL",
    "order_type": "MARKET",
    "side": "BUY",
    "quantity": 100
}
```

**Response:**
```json
{
    "order_id": "12345",
    "status": "FILLED",
    "filled_price": 150.25,
    "filled_quantity": 100,
    "timestamp": "2024-02-20T10:30:00Z"
}
```

### Analysis

#### Get Technical Indicators
```http
GET /analysis/indicators/{symbol}
```

**Parameters:**
- `symbol` (path): Stock symbol
- `indicators` (query): Comma-separated list of indicators (RSI,MA,MACD)

**Response:**
```json
{
    "symbol": "AAPL",
    "indicators": {
        "RSI": 65.5,
        "MA_20": 148.75,
        "MACD": {
            "macd": 2.5,
            "signal": 1.5,
            "histogram": 1.0
        }
    }
}
```

#### Get ML Predictions
```http
GET /analysis/predictions/{symbol}
```

**Parameters:**
- `symbol` (path): Stock symbol
- `horizon` (query): Prediction horizon in days (1, 5, 10)

**Response:**
```json
{
    "symbol": "AAPL",
    "predictions": {
        "price": 155.00,
        "confidence": 0.85,
        "timestamp": "2024-02-20T10:30:00Z"
    }
}
```

## Error Responses

All endpoints may return the following error responses:

### 400 Bad Request
```json
{
    "error": "Invalid parameters",
    "details": "Start date must be before end date"
}
```

### 401 Unauthorized
```json
{
    "error": "Unauthorized",
    "details": "Invalid API key"
}
```

### 404 Not Found
```json
{
    "error": "Not found",
    "details": "Symbol AAPL not found"
}
```

### 500 Internal Server Error
```json
{
    "error": "Internal server error",
    "details": "Database connection failed"
}
```

## Rate Limits

- 100 requests per minute for market data endpoints
- 50 requests per minute for trading endpoints
- 200 requests per minute for analysis endpoints

## WebSocket API

### Connection
```
ws://localhost:8000/api/v1/ws
```

### Subscribe to Real-time Data
```json
{
    "action": "subscribe",
    "symbols": ["AAPL", "GOOGL"],
    "channels": ["price", "volume"]
}
```

### Real-time Data Format
```json
{
    "symbol": "AAPL",
    "channel": "price",
    "data": {
        "price": 150.25,
        "timestamp": "2024-02-20T10:30:00Z"
    }
}
``` 