import streamlit as st
from langchain_openai import ChatOpenAI
from agents import DataAgent, AnalysisAgent, ReportAgent
from config import OPENAI_API_KEY, NEWS_API_KEY

# Page config
st.set_page_config(page_title="Investment Research Agent", page_icon="📈")

def main():
    st.title("📈 Investment Research Agent")
    
    # Input
    symbol = st.text_input("Enter Stock Symbol:", value="AAPL")
    
    if st.button("Analyze", type="primary"):
        if not symbol:
            st.error("Please enter a stock symbol")
            return
        
        try:
            # Initialize agents
            llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4o-mini", temperature=0)
            data_agent = DataAgent(llm)
            analysis_agent = AnalysisAgent(llm) 
            report_agent = ReportAgent(llm)
            
            # Run workflow
            with st.spinner("Collecting data..."):
                stock_data = data_agent.collect_stock_data(symbol)
                news_data = data_agent.collect_news(symbol)
            
            with st.spinner("Analyzing..."):
                analysis = analysis_agent.analyze(stock_data, news_data)
            
            with st.spinner("Generating report..."):
                report = report_agent.create_report(symbol, stock_data, analysis, news_data)
            
            # Display results
            st.success("✅ Analysis Complete!")
            
            # Summary
            st.subheader("Executive Summary")
            st.info(report['summary'])
            
            # Three columns for main info
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Price", report['data_overview']['current_price'])
                st.metric("Change", report['data_overview']['change_percent'])
                st.metric("P/E Ratio", report['data_overview']['pe_ratio'])
            
            with col2:
                st.write("**Analysis Type:**", analysis['routed_to'])
                findings = analysis['initial_analysis']
                if 'signal' in findings:
                    st.write("**Signal:**", findings['signal'])
                if 'valuation' in findings:
                    st.write("**Valuation:**", findings['valuation'])
            
            with col3:
                st.write("**News Sentiment:**", news_data['classification'])
                st.write("**Articles Found:**", news_data['news_count'])
            
            # Recommendation
            st.subheader("Recommendation")
            if 'BUY' in report['recommendation']:
                st.success(report['recommendation'])
            elif 'SELL' in report['recommendation']:
                st.error(report['recommendation'])
            else:
                st.info(report['recommendation'])
            
            # Risks
            st.subheader("Key Risks")
            for risk in report['risks']:
                st.warning(risk)
            
            # Workflow demonstration
            with st.expander("🔄 Workflow Patterns Demonstrated"):
                st.write("**1. Prompt Chaining (News Processing):**")
                st.write(f"• Classified as: {news_data['classification']}")
                st.write(f"• Extracted: {news_data['key_info'][:100]}...")
                st.write(f"• Summary: {news_data['summary']}")
                
                st.write("\n**2. Routing (Analysis):**")
                st.write(f"• Routed to: {analysis['routed_to']} specialist")
                
                st.write("\n**3. Evaluator-Optimizer:**")
                if 'evaluation' in analysis['optimized_analysis']:
                    st.write(f"• {analysis['optimized_analysis']['evaluation'][:150]}...")
                
                st.write("\n**4. Self-Reflection:**")
                st.write(f"• Data Quality: {stock_data['reflection']}")
                st.write(f"• Report Score: {report['quality_score']}/10")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()