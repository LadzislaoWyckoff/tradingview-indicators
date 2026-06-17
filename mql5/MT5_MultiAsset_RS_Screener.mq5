//+------------------------------------------------------------------+
//|                     MT5 Multi-Asset RS Screener                  |
//|                 Relative Strength vs Benchmark                    |
//|              For Forex, Indices, Commodities Analysis            |
//+------------------------------------------------------------------+
#property copyright "Trading System"
#property version   "1.00"
#property description "Scans Forex/Indices/Commodities for RS strength"

#include <Trade\Trade.mqh>

// ===== RS CALCULATION PARAMETERS =====
input int MA_LENGTH = 126;              // Moving average period
input int ZSCORE_LOOKBACK = 100;        // Z-score average length
input int LOOKBACK_DAYS = 252;          // Historical data (1 year)

// ===== WEIGHTS =====
input double ZSCORE_WEIGHT = 0.45;      // Trend weight
input double DISTANCE_WEIGHT = 0.35;    // Distance from MA weight
input double ANGLE_WEIGHT = 0.20;       // MA angle weight

// ===== ASSET GROUPS =====
string forex_pairs[] = {
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD",
    "USDCAD", "USDCHF", "EURJPY", "EURGBP", "GBPJPY"
};

string indices[] = {
    "EURUSD", "US30", "NDX", "SPX"  // Replace with actual broker symbols
};

string commodities[] = {
    "XAUUSD", "XAGUSD", "WTI", "BRENT", "NATGAS"  // Replace with actual symbols
};

// ===== BENCHMARK SYMBOLS =====
string forex_benchmark = "EURUSD";      // Main trend reference
string indices_benchmark = "SPX";       // S&P 500 for indices
string commodities_benchmark = "DXY";   // US Dollar Index for commodities

// ===== OUTPUT ARRAYS =====
struct RSResult {
    string symbol;
    double rs_score;
    string trend;
    double strength;
};

RSResult results[];

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("=" + StringFill("=", 88) + "");
    Print("MT5 RS SCREENER - MultiAsset Edition");
    Print("MA Length: ", MA_LENGTH, " | Lookback: ", LOOKBACK_DAYS, " days");
    Print("=" + StringFill("=", 88) + "");
    Print("");

    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    Print("Scanner stopped");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    // Run screener once per day
    static datetime last_run = 0;
    if(TimeCurrent() - last_run < 86400) return;  // 86400 = 1 day in seconds

    last_run = TimeCurrent();

    // Scan asset groups
    Print("\nFOREX PAIRS vs ", forex_benchmark);
    ScanAssetGroup(forex_pairs, ArraySize(forex_pairs), forex_benchmark);

    Print("\nINDICES vs ", indices_benchmark);
    ScanAssetGroup(indices, ArraySize(indices), indices_benchmark);

    Print("\nCOMMODITIES vs ", commodities_benchmark);
    ScanAssetGroup(commodities, ArraySize(commodities), commodities_benchmark);

    SaveResults();
}

//+------------------------------------------------------------------+
//| Scan asset group vs benchmark                                    |
//+------------------------------------------------------------------+
void ScanAssetGroup(string &assets[], int count, string benchmark)
{
    for(int i = 0; i < count; i++) {
        double rs = CalculateRS(assets[i], benchmark);
        if(rs != EMPTY_VALUE) {
            RSResult res;
            res.symbol = assets[i];
            res.rs_score = rs;
            res.trend = ClassifyTrend(rs);
            res.strength = CalculateStrength(rs);

            Print(StringFormat("%-10s RS=%6.1f %-20s Strength=%.0f%%",
                assets[i], rs, res.trend, res.strength*100));

            ArrayResize(results, ArraySize(results) + 1);
            results[ArraySize(results) - 1] = res;
        }
    }
}

//+------------------------------------------------------------------+
//| Calculate RS Score (3-component formula)                         |
//+------------------------------------------------------------------+
double CalculateRS(string symbol, string benchmark)
{
    // Component 1: Z-score of trend (45% weight)
    double zscore = CalculateZScoreTrend(symbol, benchmark);
    if(zscore == EMPTY_VALUE) return EMPTY_VALUE;

    // Component 2: Distance from MA (35% weight)
    double distance = CalculateDistanceFromMA(symbol);
    if(distance == EMPTY_VALUE) return EMPTY_VALUE;

    // Component 3: MA angle (20% weight)
    double angle = CalculateMAAngle(symbol);
    if(angle == EMPTY_VALUE) return EMPTY_VALUE;

    // Weighted combination
    double rs = (zscore * ZSCORE_WEIGHT) +
                (distance * DISTANCE_WEIGHT) +
                (angle * ANGLE_WEIGHT);

    return rs;
}

//+------------------------------------------------------------------+
//| Z-Score of Price Relative (symbol/benchmark)                     |
//+------------------------------------------------------------------+
double CalculateZScoreTrend(string symbol, string benchmark)
{
    int bars_needed = LOOKBACK_DAYS * 24;  // Approximate bars for daily

    double symbol_ma[];
    double benchmark_ma[];
    double ratio[];

    if(!GetMA(symbol, MA_LENGTH, bars_needed, symbol_ma)) return EMPTY_VALUE;
    if(!GetMA(benchmark, MA_LENGTH, bars_needed, benchmark_ma)) return EMPTY_VALUE;

    // Calculate price ratio
    ArrayResize(ratio, ArraySize(symbol_ma));
    for(int i = 0; i < ArraySize(symbol_ma); i++) {
        if(benchmark_ma[i] != 0)
            ratio[i] = symbol_ma[i] / benchmark_ma[i];
    }

    // Z-score of recent ratio values
    int lookback = MathMin(ZSCORE_LOOKBACK, ArraySize(ratio));
    double mean = 0, stddev = 0;

    for(int i = 0; i < lookback; i++) {
        mean += ratio[ArraySize(ratio) - 1 - i];
    }
    mean /= lookback;

    for(int i = 0; i < lookback; i++) {
        double diff = ratio[ArraySize(ratio) - 1 - i] - mean;
        stddev += diff * diff;
    }
    stddev = MathSqrt(stddev / lookback);

    if(stddev == 0) return 0;
    return (ratio[ArraySize(ratio) - 1] - mean) / stddev;
}

//+------------------------------------------------------------------+
//| Distance from Moving Average (normalized)                         |
//+------------------------------------------------------------------+
double CalculateDistanceFromMA(string symbol)
{
    double ma[];
    double close[];

    if(!GetMA(symbol, MA_LENGTH, 252, ma)) return EMPTY_VALUE;
    if(!GetClose(symbol, 252, close)) return EMPTY_VALUE;

    double current_close = close[ArraySize(close) - 1];
    double current_ma = ma[ArraySize(ma) - 1];

    if(current_ma == 0) return EMPTY_VALUE;

    return ((current_close - current_ma) / current_ma) * 100;
}

//+------------------------------------------------------------------+
//| Moving Average Angle (slope in degrees)                          |
//+------------------------------------------------------------------+
double CalculateMAAngle(string symbol)
{
    double ma[];
    if(!GetMA(symbol, MA_LENGTH, 50, ma)) return EMPTY_VALUE;

    int size = ArraySize(ma);
    if(size < 2) return EMPTY_VALUE;

    double recent_ma = ma[size - 1];
    double older_ma = ma[size - 2];

    if(recent_ma == 0) return 0;

    double slope = ((recent_ma - older_ma) / older_ma) * 100;
    return slope;
}

//+------------------------------------------------------------------+
//| Get Moving Average                                                |
//+------------------------------------------------------------------+
bool GetMA(string symbol, int period, int bars, double &result[])
{
    if(!SymbolSelect(symbol, true)) return false;

    MqlRates rates[];
    if(CopyRates(symbol, PERIOD_D1, 0, bars, rates) < 0) return false;

    ArrayResize(result, ArraySize(rates));

    double sum = 0;
    for(int i = 0; i < ArraySize(rates); i++) {
        if(i < period - 1) {
            result[i] = 0;
        } else {
            sum = 0;
            for(int j = 0; j < period; j++) {
                sum += rates[i - j].close;
            }
            result[i] = sum / period;
        }
    }

    return true;
}

//+------------------------------------------------------------------+
//| Get Close Prices                                                  |
//+------------------------------------------------------------------+
bool GetClose(string symbol, int bars, double &result[])
{
    if(!SymbolSelect(symbol, true)) return false;

    MqlRates rates[];
    if(CopyRates(symbol, PERIOD_D1, 0, bars, rates) < 0) return false;

    ArrayResize(result, ArraySize(rates));
    for(int i = 0; i < ArraySize(rates); i++) {
        result[i] = rates[i].close;
    }

    return true;
}

//+------------------------------------------------------------------+
//| Classify RS Trend                                                 |
//+------------------------------------------------------------------+
string ClassifyTrend(double rs)
{
    if(rs > 15) return "DEGRADING_STRONG";
    if(rs > 0) return "DEGRADING";
    if(rs > -15) return "OSCILLATING";
    if(rs > -30) return "IMPROVING";
    return "IMPROVING_STRONG";
}

//+------------------------------------------------------------------+
//| Calculate Trend Strength (0.0 to 1.0)                            |
//+------------------------------------------------------------------+
double CalculateStrength(double rs)
{
    double abs_rs = MathAbs(rs);
    if(abs_rs < 15) return 0.25;
    if(abs_rs < 30) return 0.50;
    if(abs_rs < 50) return 0.75;
    return 1.00;
}

//+------------------------------------------------------------------+
//| Save Results to File                                              |
//+------------------------------------------------------------------+
void SaveResults()
{
    string filename = "rs_screener_results_" + TimeToString(TimeCurrent(), TIME_DATE) + ".csv";
    int handle = FileOpen(filename, FILE_WRITE | FILE_CSV);

    if(handle == INVALID_HANDLE) return;

    FileWrite(handle, "Symbol,RS_Score,Trend,Strength");
    for(int i = 0; i < ArraySize(results); i++) {
        FileWrite(handle,
            results[i].symbol,
            DoubleToString(results[i].rs_score, 1),
            results[i].trend,
            DoubleToString(results[i].strength * 100, 1));
    }

    FileClose(handle);
    Print("Results saved to: ", filename);
}

//+------------------------------------------------------------------+
//| Utility: String filler                                            |
//+------------------------------------------------------------------+
string StringFill(string str, int count)
{
    string result = "";
    for(int i = 0; i < count; i++) {
        result += str;
    }
    return result;
}
