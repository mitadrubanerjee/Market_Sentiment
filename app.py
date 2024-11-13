import streamlit as st
import requests
import yfinance as yf
import time
import pandas as pd
import plotly.express as px
from openai import OpenAI # type: ignore
from scoring_script.scoring import make_prediction
from services.result_rationalizer import rationalize_result
#from result_rationalizer import rationalize_result

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure VADER lexicon is downloaded
nltk.download('vader_lexicon')

API_Key = st.secrets["openai_api_key"]
secret_key = API_Key

# Function to generate a summary of headlines using OpenAI Chat API
client = OpenAI(api_key=secret_key)

# Function to generate a summary of headlines using OpenAI's updated API
def generate_summary(headlines):
    # Join all headlines into a single prompt string
    prompt = "Hi, imagine that you are a very respected and very experienced journalist. So, there are a lot of news that we get everyday right. It can sometime be very overwhelming to go through all the news headlines individually. Its is also important to keep its simple and extract the main theme from all the ews articles while you are reading it. Always keep in mind the context, s that will make the summary much more grounded. So, I would want you to summarize all the news that is here into a four-line summary (preferably in bullet points):\n\n" + "\n".join(headlines)
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Imagine that you are a very respected and very experienced journalist with over 30 years of experince in journalism. It is important to keep in mind that the headlines need to summarized in a clear concise way, so the reader can just understand all the main point by just glancing over it. Also another thing to consider, maybe just consider the historical context of the news you are summarizing, so when you are summarizing something, keep all the past context (as far as you know) in mind. Also see to it that you do not give any incomplete responses please. "},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0.2
    )
    
    # Retrieve the summary from the response
    #summary = response.choices[0].message['content'].strip()
    return response.choices[0].message.content.strip() 
    #return summary

def analyze_sentiment(descriptions):
    prompt = (
        "You are an expert in sentiment analysis. Analyze the overall sentiment "
        "of the following news article descriptions and provide any of the following sentiments: "
        "Positive, Trending Positive, Trending Negative, Negative, or Neutral. "
        "So, if the sentiment is definitiely Positive/Negative then either respond Positive or Negative, but if it is getting hard to decide "
        "and if you think its between Negative and Neutral then respond with Trending Negative. If its between Positive and Neutral then respond with Trending Positive"
        "If its genuinely Neutral, then just respond with Neutral"
        "Think hard before giving something outright Negative. If something is sounding Positive, maybe go with it"
        "Always provide one of the responses, never give a blank response.\n\n"
        f"{descriptions}"
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert in sentiment analysis of news articles."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=10,
        temperature=0.2
    )

    return response.choices[0].message.content.strip() 

# Inject CSS for card styling and marquee effect
st.markdown("""
    <style>
    /* Card styling */
    .card {
        border: 1px solid #ccc;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        background-color: #f9f9f9;
    }
    .card h4 {
        margin: 0;
        color: #1e90ff;
    }
    .card p {
        color: #444;
    }
    .card .description {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        color: #333;
        max-width: 100%;
    }
    .card a {
        color: #1e90ff;
        text-decoration: none;
    }
    .card a:hover {
        text-decoration: underline;
    }
        /* Centering the title */
    .centered-title {
        text-align: center;

    </style>
    """, unsafe_allow_html=True)

# Centered title
st.markdown(
    """
    <h1 class='centered-title'>
        📰 Market Sentiment Summarizer & Predictor
    </h1>
    """,
    unsafe_allow_html=True
)

# Function to fetch news articles from Bing Search API
def get_news_articles(query):
    api_key = st.secrets["bing_api_key"]  
    news_search_endpoint = "https://api.bing.microsoft.com/v7.0/news/search"
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    all_articles = []
    offset = 0

    while offset < 150:
        params = {
            "q": query,
            "count": 100,
            "offset": offset,
            "freshness": "Month",
            "mkt": "en-US",
        }
        response = requests.get(news_search_endpoint, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        articles = data.get("value", [])
        if not articles:
            break
        all_articles.extend(articles)
        offset += 100
        time.sleep(1)
    
    unique_articles = {article['url']: article for article in all_articles}.values()
    articles_data = [
        {
            "Title": article.get("name"),
            "Description": article.get("description"),
            "URL": article.get("url"),
            "Published At": article.get("datePublished"),
            "Provider": article.get("provider")[0].get("name") if article.get("provider") else "N/A",
        }
        for article in unique_articles
    ]
    return pd.DataFrame(articles_data)


def display_articles_in_grid(articles):
    top_articles = articles.head(20)
    cols = st.columns(3)  # Three-column layout
    for idx, (_, row) in enumerate(top_articles.iterrows()):
        with cols[idx % 3]:  # Rotate through columns
            st.markdown(f"""
                <div style="
                    border: 1px solid #444; 
                    border-radius: 10px; 
                    padding: 15px; 
                    margin: 10px 0; 
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.6); 
                    background-color: #333; 
                    color: #f0f0f0;
                ">
                    <h4 style="margin-bottom: 10px;">
                        <a href="{row['URL']}" target="_blank" style="color:#1e90ff; text-decoration:none;">
                            {row['Title']}
                        </a>
                    </h4>
                </div>
            """, unsafe_allow_html=True)



# Streamlit app layout
#st.title("📰 News Summarizer & Sentiment Analyzer")

# Step 1: User choice selection
choice = st.radio(
    "Choose your analysis type:",
    ("General Sentiment of a Topic", "Company-Specific Sentiment and Prediction")
)

# Step 2: General Sentiment Workflow
if choice == "General Sentiment of a Topic":
    # Input field for keyword (General topic)
    keyword = st.text_input("Enter Topic/Keyword for General Sentiment Analysis")
    
    if st.button("Fetch Articles for General Sentiment"):
        if keyword:
            with st.spinner("Fetching Articles, Generating Summary and Sentiments"):
                progress_bar = st.progress(0)
                articles_df = get_news_articles(keyword)

                for percent_complete in range(0, 101, 10):
                    time.sleep(0.1)
                    progress_bar.progress(percent_complete)

            st.write(f"Total articles found: {len(articles_df)}")

            # Add artificial delay and progress updates
            for percent_complete in range(0, 101, 10):
                time.sleep(0.1)
                progress_bar.progress(percent_complete)

            # Generate summary of headlines
            headlines = articles_df['Title'].tolist()
            summary = generate_summary(headlines)

            # Display the summary at the top
            st.write("### Summary of Headlines")
            st.info(summary)

            # Concatenate all descriptions for sentiment analysis
            all_descriptions = " ".join(articles_df['Description'].dropna().tolist())
            overall_sentiment = analyze_sentiment(all_descriptions)

            # Display the overall sentiment
            st.write("### Overall Sentiment of Articles")
            st.success(f"The overall sentiment of the news articles is: **{overall_sentiment}**")
            
            # Display articles in grid layout
            st.write("### Latest News Articles")
            display_articles_in_grid(articles_df)
        else:
            st.warning("Please enter a topic or keyword for sentiment analysis.")

# Step 3: Company-Specific Sentiment and Prediction Workflow
elif choice == "Company-Specific Sentiment and Prediction":
    # Input fields for Company Name and Ticker
    company_name = st.text_input("Enter the company name for prediction:")
    ticker_symbol = st.text_input("Enter the stock ticker symbol for prediction:")
    
    if st.button("Fetch Articles and Predict"):
        if company_name and ticker_symbol:
            with st.spinner("Fetching Articles, Generating Summary, Sentiments, and Prediction"):
                # Fetch articles related to the company
                articles_df = get_news_articles(company_name)
                progress_bar = st.progress(0)
                
                # Simulate progress for article fetching
                for percent_complete in range(0, 101, 10):
                    time.sleep(0.1)
                    progress_bar.progress(percent_complete)

            st.write(f"Total articles found: {len(articles_df)}")

            # Generate summary of headlines
            headlines = articles_df['Title'].tolist()
            summary = generate_summary(headlines)

            # Display the summary at the top
            st.write("### Summary of Headlines")
            st.info(summary)

            # Concatenate all descriptions for sentiment analysis
            all_descriptions = " ".join(articles_df['Description'].dropna().tolist())
            overall_sentiment = analyze_sentiment(all_descriptions)

            # Display the overall sentiment
            st.write("### Overall Sentiment of Articles")
            st.success(f"The overall sentiment of the news articles is: **{overall_sentiment}**")


            # Get the prediction from scoring.py
            prediction = make_prediction(company_name, ticker_symbol)
            
            # Rationalize prediction and sentiment
            final_message = rationalize_result(overall_sentiment, prediction)
            
            # Display the final rationalized message
            st.subheader("Prediction Result")
            st.success(final_message)

                        # *********** Added Stock Performance Chart ***********
            # Fetch and display stock performance chart for the last month
            st.write(f"### {company_name} ({ticker_symbol}) Stock Performance - Last Month")
            
            # Fetch stock data for the past month
            stock_data = yf.download(ticker_symbol, period="1mo", interval="1d")

            # Alternatively, use Plotly for an interactive chart
            # Ensure `stock_data['Close']` is one-dimensional
            # Ensure the DataFrame has a Date column to avoid indexing issues
            # Ensure the DataFrame is single-indexed
            stock_data = stock_data.reset_index()

            # If there is a multi-index in columns, flatten it
            if isinstance(stock_data.columns, pd.MultiIndex):
                stock_data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in stock_data.columns]

            # Rename the date column to ensure it's accessible as 'Date'
            date_columns = [col for col in stock_data.columns if 'Date' in col or 'index' in col]
            if date_columns:
                stock_data = stock_data.rename(columns={date_columns[0]: 'Date'})
            else:
                st.error("Unable to locate the 'Date' column in the data.")
                st.stop()

            # Dynamically find and rename the Close column
            close_columns = [col for col in stock_data.columns if 'Close' in col]
            if close_columns:
                stock_data = stock_data.rename(columns={close_columns[0]: 'Close'})
            else:
                st.error("Unable to locate the 'Close' column in the data.")
                st.stop()

            # Repeat similar logic for Open, High, and Low if necessary
            open_columns = [col for col in stock_data.columns if 'Open' in col]
            if open_columns:
                stock_data = stock_data.rename(columns={open_columns[0]: 'Open'})

            high_columns = [col for col in stock_data.columns if 'High' in col]
            if high_columns:
                stock_data = stock_data.rename(columns={high_columns[0]: 'High'})

            low_columns = [col for col in stock_data.columns if 'Low' in col]
            if low_columns:
                stock_data = stock_data.rename(columns={low_columns[0]: 'Low'})

            # Flatten the Close column to ensure it's 1D
            stock_data['Close'] = stock_data['Close'].squeeze()

            # Using Plotly for an interactive chart
            import plotly.express as px

            # Plot the line chart with Date and Close columns
            fig = px.line(stock_data, x='Date', y='Close', title=f"{company_name} Stock Price Over Last Month")
            st.plotly_chart(fig)

            # Optional: Add candlestick chart if 'Open', 'High', 'Low' columns are available
            if 'Open' in stock_data.columns and 'High' in stock_data.columns and 'Low' in stock_data.columns and 'Close' in stock_data.columns:
                import plotly.graph_objects as go
                fig = go.Figure(
                    data=[
                        go.Candlestick(
                            x=stock_data['Date'],
                            open=stock_data['Open'],
                            high=stock_data['High'],
                            low=stock_data['Low'],
                            close=stock_data['Close'],
                            name="Candlestick"
                        )
                    ]
                )
                fig.update_layout(title=f"{company_name} Stock Price Candlestick Chart - Last Month", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig)


            
            # *********** End of Stock Performance Chart ***********
            # Display articles in grid layout
            st.write("### Latest News Articles")
            display_articles_in_grid(articles_df)
        
        else:
            st.warning("Please enter both the company name and stock ticker symbol.")