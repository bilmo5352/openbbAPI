# Trading Analysis Backend API

FastAPI-based trading analysis service that fetches stock data from Zerodha KiteConnect API (NSE/BSE) or yfinance fallback, and computes technical indicators using OpenBB extensions.

## Features

- **Data Sources**: Zerodha KiteConnect (NSE/BSE) with yfinance fallback
- **Technical Indicators**: 200+ indicators from multiple libraries
  - **OpenBB Technical Extension**: ~150-200 indicators
  - **TA-Lib**: ~150 indicators (industry standard)
  - **pandas-ta**: ~200+ indicators (most comprehensive)
  - **Manual implementations**: Core indicators (always available)
- **Multi-Library Support**: Automatically uses best available library (OpenBB > TA-Lib > pandas-ta > manual)
- **RESTful API**: FastAPI with automatic OpenAPI documentation

## API Endpoints

### Health Check
```
GET /health
```
Returns: `{"status": "ok"}`

### Analyze Ticker
```
POST /analyze
```

**Request Body:**
```json
{
  "ticker": "RELIANCE",
  "start": "2024-11-01",
  "end": "2024-12-10",
  "exchange": "NSE",
  "use_yfinance_fallback": true,
  "indicators": ["rsi", "atr", "vwap", "ichimoku", "macd"]
}
```

**Response:**
```json
{
  "ticker": "RELIANCE",
  "exchange": "NSE",
  "start": "2024-11-01",
  "end": "2024-12-10",
  "profile": {...},
  "rows": 237,
  "data": [...],
  "computed_indicators": ["rsi", "atr", "vwap", "ichimoku", "macd"],
  "skipped_indicators": []
}
```

## Available Indicators

The API supports **200+ technical indicators** from three major libraries. Check available indicators via:

```bash
GET /indicators
```

### Indicator Categories

**Trend Indicators:**
- Moving Averages: `sma`, `ema`, `dema`, `tema`, `wma`, `hma`, `kama`, `t3`, `trima`, `zlma`, `alma`, `jma`, `vidya`
- Trend Analysis: `adx`, `adxr`, `aroon`, `aroonosc`, `dx`, `psar`, `supertrend`, `ichimoku`, `vortex`

**Momentum Indicators:**
- Oscillators: `rsi`, `stoch`, `stochrsi`, `stochf`, `willr`, `cci`, `cmo`, `mfi`, `roc`, `mom`, `ppo`, `trix`, `ultosc`
- Advanced: `fisher`, `ao`, `cg`, `kst`, `tsi`, `uo`, `smi`, `qqe`, `inertia`, `squeeze`, `squeeze_pro`

**Volatility Indicators:**
- Bands: `bbands`, `kc` (Keltner Channels), `donchian`, `accbands`, `aberration`
- Volatility: `atr`, `natr`, `trange`, `true_range`, `rvi`, `ui`, `massi`, `thermo`

**Volume Indicators:**
- Volume Analysis: `obv`, `ad`, `adosc`, `cmf`, `eom`, `efi`, `kvo`, `mfi`, `pvi`, `nvi`, `aobv`
- Volume-Weighted: `vwap`, `vwma`, `fwma`

**Pattern Recognition:**
- 60+ Candlestick Patterns: `cdl2crows`, `cdl3blackcrows`, `cdlengulfing`, `cdlhammer`, `cdlharami`, `cdlmorningstar`, `cdleveningstar`, and many more

**Price Transform:**
- Price Calculations: `typprice`, `wclprice`, `avgprice`, `medprice`, `hl2`, `hlc3`, `ohlc4`, `wcp`

**Other Indicators:**
- Fibonacci: `fib`
- Demark: `demark`
- Relative Rotation: `relative_rotation`
- Clenow: `clenow`
- Cones: `cones`

### Library Priority

Indicators are computed using the first available library in this order:
1. **OpenBB Technical Extension** (if installed)
2. **TA-Lib** (if installed)
3. **pandas-ta** (if installed)
4. **Manual implementation** (always available for core indicators)

## Local Development

### Prerequisites
- Python 3.11+
- pip

### Installation

1. Clone the repository and navigate to BBbackend:
```bash
cd BBbackend
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

**Optional: Install Technical Analysis Libraries**

For maximum indicator coverage, install additional libraries:

```bash
# pandas-ta (200+ indicators) - Recommended
pip install pandas-ta

# TA-Lib (150 indicators) - Requires C library first
# Windows: Download wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
#          Then: pip install TA_Lib-0.4.XX-cpXX-cpXX-win_amd64.whl
# Linux:   sudo apt-get install ta-lib && pip install TA-Lib
# macOS:   brew install ta-lib && pip install TA-Lib
pip install TA-Lib

# OpenBB Technical Extension (150-200 indicators)
pip install openbb[technical]
```

**Note:** The API works without these libraries - it will gracefully fall back to available implementations. Manual implementations are always available for core indicators.

4. Set environment variables (optional):
```bash
cp .env.example .env
# Edit .env with your Zerodha credentials
```

5. Run the server:
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

6. Access the API:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

### Testing the API

Use the included client:
```bash
python client.py --ticker RELIANCE --start 2024-11-01 --end 2024-12-10 --exchange NSE
```

Or use curl:
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "RELIANCE",
    "start": "2024-11-01",
    "end": "2024-12-10",
    "exchange": "NSE",
    "indicators": ["rsi", "macd"]
  }'
```

## Railway Deployment

### Option 1: Deploy via Railway CLI

1. Install Railway CLI:
```bash
npm i -g @railway/cli
```

2. Login to Railway:
```bash
railway login
```

3. Initialize and deploy:
```bash
cd BBbackend
railway init
railway up
```

4. Set environment variables in Railway dashboard:
   - `KITE_API_KEY` (optional)
   - `KITE_ACCESS_TOKEN` (optional)
   - `PORT` is automatically set by Railway

### Option 2: Deploy via Railway Dashboard

1. Go to [Railway](https://railway.app)
2. Create a new project
3. Connect your GitHub repository
4. Select the `BBbackend` directory as the root
5. Railway will automatically detect the Dockerfile
6. Add environment variables in the Variables tab
7. Deploy!

### Option 3: Deploy with Docker

1. Build the Docker image:
```bash
docker build -t trading-api .
```

2. Run the container:
```bash
docker run -p 8000:8000 \
  -e KITE_API_KEY=your_key \
  -e KITE_ACCESS_TOKEN=your_token \
  trading-api
```

## Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `KITE_API_KEY` | Zerodha KiteConnect API key | No | Uses test credentials |
| `KITE_ACCESS_TOKEN` | Zerodha KiteConnect access token | No | Uses test credentials |
| `PORT` | Server port | No | 8000 (Railway sets automatically) |

## Project Structure

```
BBbackend/
├── api.py                 # FastAPI application
├── analysis_service.py    # Business logic for analysis
├── main.py                # Data fetching and indicator computation
├── client.py              # Test client (not needed for deployment)
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker configuration
├── railway.json           # Railway deployment config
├── .env.example           # Environment variables template
└── README.md             # This file
```

## API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Troubleshooting

### Indicator Library Installation

**OpenBB Installation Issues:**
- If OpenBB fails to install, the API will fall back to TA-Lib, pandas-ta, or manual implementations
- This is handled gracefully - the API works with any combination of libraries

**TA-Lib Installation Issues:**
- TA-Lib requires the C library to be installed first
- If TA-Lib is not available, indicators will use pandas-ta or manual implementations
- Check installation: `python -c "import talib; print(talib.__version__)"`

**pandas-ta Installation:**
- Usually installs without issues: `pip install pandas-ta`
- If unavailable, core indicators will use manual implementations

**Checking Available Libraries:**
```bash
# Check which libraries are available
curl http://localhost:8000/indicators
```

The response includes:
- `openbb_technical_available`: Whether OpenBB is installed
- `talib_available`: Whether TA-Lib is installed
- `pandas_ta_available`: Whether pandas-ta is installed
- `total_supported`: Total indicators in catalog
- `total_available`: Indicators available with current libraries

### Zerodha Connection Issues
If Zerodha API fails, set `use_yfinance_fallback: true` in your request to use yfinance as fallback.

### Port Issues on Railway
Railway automatically sets the `PORT` environment variable. The Dockerfile and railway.json handle this automatically.

## License

This project is for educational and personal use.

