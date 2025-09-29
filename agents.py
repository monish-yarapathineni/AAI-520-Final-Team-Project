import yfinance as yf
import requests
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
from config import NEWS_API_KEY

class DataAgent:
    """Data Agent : Collects data"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def collect_stock_data(self, symbol):
        """Get stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            hist = stock.history(period="5d")
            
            price = round(hist['Close'].iloc[-1], 2) if not hist.empty else 0
            change = round(((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100, 2) if len(hist) > 1 else 0
            
            data = {
                "symbol": symbol,
                "price": price,
                "change": change,
                "volume": int(hist['Volume'].iloc[-1]) if not hist.empty else 0,
                "pe_ratio": round(info.get('trailingPE', 0), 2),
                "market_cap": info.get('marketCap', 0)
            }
            
            # Self-reflection
            missing = [k for k, v in data.items() if v == 0 or v == 'N/A']
            if not missing:
                reflection = "✅ All data collected successfully"
            else:
                reflection = f"⚠️ Some data missing: {', '.join(missing[:3])}"
            
            data['reflection'] = reflection
            return data
            
        except:
            return {
                "symbol": symbol, 
                "price": 0, 
                "change": 0, 
                "volume": 0, 
                "pe_ratio": 0, 
                "market_cap": 0,
                "reflection": "❌ Error collecting data"
            }
    
    def collect_news(self, symbol):
        """Get news and process with prompt chaining"""
        # Fetch news
        news_text = self.fetch_news(symbol)
        
        # Classify (Prompt Chain Step 1)
        classify_prompt = f"Classify this news sentiment as positive, negative, or neutral: {news_text}"
        classification = self.llm.invoke(classify_prompt).content
        
        # Extract (Prompt Chain Step 2)  
        extract_prompt = f"Extract the key point from: {news_text}"
        key_info = self.llm.invoke(extract_prompt).content
        
        # Summarize (Prompt Chain Step 3)
        summary_prompt = f"In one sentence, market impact of: {key_info}"
        summary = self.llm.invoke(summary_prompt).content
        
        return {
            "news": news_text[:100],
            "classification": classification,
            "key_info": key_info,
            "summary": summary,
            "news_count": 1
        }
    
    def fetch_news(self, symbol):
        """Fetch news from NewsAPI"""
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': symbol,
                'from': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                'sortBy': 'relevancy',
                'pageSize': 1,
                'apiKey': NEWS_API_KEY
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data.get('articles'):
                    return data['articles'][0].get('title', f"{symbol} shows growth")
        except:
            print("Error fetching news")
        return f"{symbol} reports strong quarterly earnings and positive outlook"


class AnalysisAgent:
    """Analysis Agent: Analyzes with routing"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def analyze(self, stock_data, news_data):
        """Route to appropriate analysis"""
        # Routing decision
        if stock_data.get('pe_ratio', 0) > 0:
            route = "fundamental"
            analysis = self.fundamental_analysis(stock_data)
        else:
            route = "technical"
            analysis = self.technical_analysis(stock_data)
        
        # Add news sentiment
        analysis["news_sentiment"] = news_data.get("classification", "neutral")
        
        # Evaluate and optimize
        evaluation = self.evaluate(analysis)
        
        return {
            "routed_to": route,
            "initial_analysis": analysis,
            "optimized_analysis": {"evaluation": evaluation}
        }
    
    def technical_analysis(self, data):
        """Technical analysis based on price movement"""
        change = data.get('change', 0)
        
        if change > 2:
            signal = "BUY - Positive momentum"
        elif change < -2:
            signal = "SELL - Negative momentum"
        else:
            signal = "HOLD - Neutral"
        
        return {"type": "technical", "signal": signal, "change": change}
    
    def fundamental_analysis(self, data):
        """Fundamental analysis based on PE ratio"""
        pe = data.get('pe_ratio', 0)
        
        if 0 < pe < 15:
            valuation = "Undervalued"
        elif pe > 30:
            valuation = "Overvalued"
        else:
            valuation = "Fair value"
        
        return {"type": "fundamental", "valuation": valuation, "pe_ratio": pe}
    
    def evaluate(self, analysis):
        """Evaluate and optimize analysis"""
        prompt = f"Rate this analysis quality (1-10) and suggest improvement: {analysis}"
        return self.llm.invoke(prompt).content[:200]


class ReportAgent:
    """Report Agent: Creates report"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def create_report(self, symbol, stock_data, analysis, news_data):
        """Generate investment report"""
        # Create report with consistent structure
        report = {
            "symbol": symbol,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "summary": self.create_summary(stock_data, analysis, news_data),
            "data_overview": {
                "current_price": f"${stock_data.get('price', 0)}",
                "change_percent": f"{stock_data.get('change', 0)}%",
                "pe_ratio": stock_data.get('pe_ratio', 'N/A'),
                "volume": f"{stock_data.get('volume', 0):,}" if stock_data.get('volume') else "0"
            },
            "recommendation": self.create_recommendation(analysis, news_data),
            "risks": self.identify_risks(symbol),
        }
        # Add quality score after creating report
        report["quality_score"] = self.self_evaluate(report)
        
        return report
    
    def create_summary(self, stock_data, analysis, news_data):
        """Create executive summary"""
        prompt = f"""
        Write 2-sentence summary for {stock_data.get('symbol')}:
        Price change: {stock_data.get('change')}%
        Analysis: {analysis.get('initial_analysis')}
        News: {news_data.get('classification')}
        """
        return self.llm.invoke(prompt).content
    
    def create_recommendation(self, analysis, news_data):
        """Generate recommendation"""
        analysis_detail = analysis.get('initial_analysis', {})
        sentiment = news_data.get('classification', 'neutral')
        
        prompt = f"""
        Based on {analysis_detail} and news sentiment {sentiment},
        give recommendation (BUY/HOLD/SELL) with one-line reason.
        """
        return self.llm.invoke(prompt).content
    
    def identify_risks(self, symbol):
        """Identify 2 risks"""
        prompt = f"List 2 main investment risks for {symbol} in one line each."
        risks = self.llm.invoke(prompt).content.split('\n')
        return [r.strip() for r in risks[:2] if r.strip()]
    
    def self_evaluate(self, report):
        """Self-reflection on report quality"""
        score = 0
        
        # Check if summary exists and is meaningful
        if report.get('summary') and len(report['summary']) > 50:
            score += 3
        
        # Check if recommendation is clear
        recommendation = report.get('recommendation', '')
        if 'BUY' in recommendation or 'SELL' in recommendation or 'HOLD' in recommendation:
            score += 3
        
        # Check if risks are identified
        if report.get('risks') and len(report['risks']) >= 2:
            score += 2
        
        # Check if data is complete
        if report.get('data_overview', {}).get('current_price') != '$0':
            score += 2
    
        return min(score, 10)  # Max score 10


class OrchestratorAgent:
    """Orchestrator Agent: Coordinates all other agents"""
    
    def __init__(self, llm):
        self.llm = llm
        self.data_agent = DataAgent(llm)
        self.analysis_agent = AnalysisAgent(llm)
        self.report_agent = ReportAgent(llm)
    
    def plan_research(self, symbol):
        """Plan the research workflow"""
        plan_prompt = f"""
        Create a research plan for {symbol} with these steps:
        1. What data to collect
        2. What analysis to perform
        3. What to include in report
        Return as numbered list.
        """
        plan = self.llm.invoke(plan_prompt).content
        return plan.split('\n')
    
    def execute_research(self, symbol):
        """Orchestrate the entire research process"""
        # Plan the workflow
        workflow_plan = self.plan_research(symbol)
        
        # Execute data collection
        stock_data = self.data_agent.collect_stock_data(symbol)
        news_data = self.data_agent.collect_news(symbol)
        
        # Perform analysis
        analysis = self.analysis_agent.analyze(stock_data, news_data)
        
        #  Generate report
        report = self.report_agent.create_report(symbol, stock_data, analysis, news_data)
        
        #  Self-reflect on entire workflow
        reflection = self.reflect_on_workflow(workflow_plan, report)
        
        return {
            "plan": workflow_plan,
            "stock_data": stock_data,
            "news_data": news_data,
            "analysis": analysis,
            "report": report,
            "reflection": reflection
        }
    
    def reflect_on_workflow(self, plan, report):
        """Reflect on the entire workflow execution"""
        prompt = f"""
        Evaluate workflow execution:
        Plan: {plan[:3]}
        Report quality: {report.get('quality_score')}/10
        What went well and what could improve?
        """
        return self.llm.invoke(prompt).content[:200]