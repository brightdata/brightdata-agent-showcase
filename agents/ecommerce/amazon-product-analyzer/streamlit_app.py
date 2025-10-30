import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import math
from datetime import datetime, timezone

from amazon_analytics.api import BrightDataAPI
from amazon_analytics.data_processor import DataProcessor
from amazon_analytics.shopping_intelligence import ShoppingIntelligenceEngine
from amazon_analytics.gemini_ai_engine import get_gemini_ai

st.set_page_config(
    page_title="Smart Shopping Assistant",
    page_icon="🛍️",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: #f8f9fa;
        border-bottom: 1px solid #e9ecef;
        margin-bottom: 2rem;
    }
    
    .main-title {
        font-size: 1.6rem;
        font-weight: 500;
        color: #2c3e50;
        margin: 0;
    }
    
    .subtitle {
        color: #6c757d;
        font-size: 0.9rem;
        font-weight: 400;
        margin-top: 0.5rem;
    }
    
    /* Clean typography */
    h1, h2, h3, h4 {
        color: #2c3e50;
        font-weight: 500;
    }
    
    h4 {
        font-size: 1.1rem;
        margin-bottom: 1rem;
    }
    
    /* Minimal button styling */
    .stButton > button {
        background: #0d6efd;
        color: white;
        border: none;
        border-radius: 4px;
        font-weight: 400;
    }
    
    .stButton > button:hover {
        background: #0b5ed7;
    }
    
    /* Clean input styling with consistent focus */
    .stTextInput > div > div > input {
        border-radius: 4px;
        border: 1px solid #ced4da;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #0d6efd !important;
        box-shadow: 0 0 0 2px rgba(13, 110, 253, 0.25) !important;
        outline: none !important;
    }
    
    /* Override browser focus styles for all elements */
    .stSelectbox > div > div > div:focus,
    .stTextArea > div > div > textarea:focus,
    input:focus,
    textarea:focus,
    select:focus,
    button:focus {
        outline: none !important;
        border-color: #0d6efd !important;
        box-shadow: 0 0 0 2px rgba(13, 110, 253, 0.25) !important;
    }
    
    /* Remove unnecessary visual elements */
    .stExpander {
        border: 1px solid #e9ecef;
        border-radius: 4px;
    }
    
    /* Mobile responsive design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 1.3rem;
        }
        
        .main-header {
            padding: 1rem 0;
            margin-bottom: 1rem;
        }
        
        /* Better mobile metrics display */
        div[data-testid="metric-container"] {
            margin-bottom: 0.5rem;
        }
        
        /* Mobile-friendly expanders */
        .stExpander {
            margin-bottom: 0.5rem;
        }
        
        /* Responsive button sizing */
        .stButton > button {
            font-size: 0.9rem;
            padding: 0.5rem 1rem;
        }
    }
    
    /* Tablet responsive design */
    @media (max-width: 1024px) and (min-width: 769px) {
        .main-title {
            font-size: 1.5rem;
        }
    }
    
    /* Enhanced product image styling */
    img {
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'shopping_intelligence' not in st.session_state:
    st.session_state.shopping_intelligence = {}
if 'current_run_id' not in st.session_state:
    st.session_state.current_run_id = None

@st.cache_resource
def get_backend_components():
    """Initialize and cache backend components."""
    api = BrightDataAPI()
    processor = DataProcessor()
    intelligence = ShoppingIntelligenceEngine()
    ai_engine = get_gemini_ai()
    return api, processor, intelligence, ai_engine

if 'backend_reload' not in st.session_state:
    st.cache_resource.clear()
    st.session_state.backend_reload = True

api, processor, intelligence, ai_engine = get_backend_components()

st.markdown("""
<div class="main-header">
    <h1 class="main-title">🛍️ Amazon Product Analytics</h1>
    <p class="subtitle">Advanced product discovery and market analysis</p>
    <p style="font-size: 14px; color: #888; margin-top: 8px;">Powered by <a href="http://brightdata.com/" target="_blank" style="color: #4A90E2; text-decoration: none; font-weight: 500;">Bright Data</a></p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    countries = {
        "🇺🇸 United States": "US",
        "🇨🇦 Canada": "CA", 
        "🇲🇽 Mexico": "MX",
        "🇬🇧 United Kingdom": "GB",
        "🇩🇪 Germany": "DE",
        "🇫🇷 France": "FR",
        "🇮🇹 Italy": "IT",
        "🇪🇸 Spain": "ES",
        "🇳🇱 Netherlands": "NL",
        "🇸🇪 Sweden": "SE",
        "🇵🇱 Poland": "PL",
        "🇧🇷 Brazil": "BR",
        "🇦🇺 Australia": "AU",
        "🇯🇵 Japan": "JP",
        "🇮🇳 India": "IN",
        "🇸🇬 Singapore": "SG",
        "🇹🇷 Turkey": "TR",
        "🇸🇦 Saudi Arabia": "SA",
        "🇦🇪 United Arab Emirates": "AE",
        "🇿🇦 South Africa": "ZA"
    }
    
    currencies = {
        "US": "USD", "CA": "CAD", "MX": "MXN", "GB": "GBP", "DE": "EUR", 
        "FR": "EUR", "IT": "EUR", "ES": "EUR", "NL": "EUR", "SE": "SEK", 
        "PL": "PLN", "BR": "BRL", "AU": "AUD", "JP": "JPY", "IN": "INR", 
        "SG": "SGD", "TR": "TRY", "SA": "SAR", "AE": "AED", "ZA": "ZAR"
    }
    
    currency_symbols = {
        "USD": "$", "CAD": "C$", "MXN": "$", "GBP": "£", "EUR": "€", 
        "SEK": "kr", "PLN": "zł", "BRL": "R$", "AUD": "A$", "JPY": "¥", 
        "INR": "₹", "SGD": "S$", "TRY": "₺", "SAR": "﷼", "AED": "د.إ", "ZAR": "R"
    }
    
    default_country = st.session_state.get('selected_country', list(countries.keys())[0])
    default_index = list(countries.keys()).index(default_country) if default_country in countries else 0
    
    selected_country = st.selectbox(
        "🌍 Select Your Market",
        options=list(countries.keys()),
        index=default_index
    )
    
    st.session_state.selected_country = selected_country
    
    keyword = st.text_input(
        "🔍 What product are you looking for?",
        placeholder="e.g., wireless headphones, coffee maker, iPhone 15..."
    )
    
    st.markdown("**💡 Popular Searches:**")
    popular_searches = [
        ("🎧", "headphones"), ("📱", "iPhone 15"), ("💻", "laptop"), 
        ("☕", "coffee maker"), ("👟", "running shoes"), ("📺", "smart TV")
    ]
    
    for row in range(2):
        cols = st.columns(3)
        for col_idx in range(3):
            search_idx = row * 3 + col_idx
            if search_idx < len(popular_searches):
                emoji, term = popular_searches[search_idx]
                with cols[col_idx]:
                    if st.button(f"{emoji} {term}", key=f"popular_{search_idx}", 
                               help=f"Search for {term}", use_container_width=True):
                        st.session_state.popular_search_clicked = term
                        st.rerun()
    
    if hasattr(st.session_state, 'popular_search_clicked'):
        keyword = st.session_state.popular_search_clicked
        del st.session_state.popular_search_clicked
        search_clicked = True
    else:
        search_clicked = st.button("🔍 Search & Analyze", type="primary", use_container_width=True, help="Find products and analyze market data")

def format_price(amount, country_code):
    """Format price with appropriate currency symbol.
    
    Args:
        amount (float): Price amount to format
        country_code (str): Two-letter country code (e.g., 'US', 'GB', 'DE')
        
    Returns:
        str: Formatted price string with currency symbol
    """
    currency_code = currencies.get(country_code, "USD")
    symbol = currency_symbols.get(currency_code, "$")
    
    # Localized currency display with regional formatting conventions
    if currency_code == "JPY":
        return f"{symbol}{amount:,.0f}"  # No decimals for Yen
    elif currency_code in ["SEK", "PLN"]:
        return f"{amount:,.2f} {symbol}"  # Symbol after amount for some currencies
    else:
        return f"{symbol}{amount:,.2f}"  # Standard format

def smart_wait_for_results(api, snapshot_id, progress_bar, status_text, time_counter, start_time):
    """Enhanced wait function with real-time feedback and smart messaging."""
    import time
    
    wait_messages = [
        (0, "⏳ Amazon is processing your search..."),
        (30, "🔍 Still searching... Amazon servers are working hard!"),
        (60, "⏱️ Taking longer than usual - complex search in progress..."),
        (120, "🕑 Sorry for the wait! Amazon has a lot of products to check..."),
        (180, "⏳ Almost there! Quality results take time..."),
        (300, "🔄 Extended search - Amazon is being thorough with your request...")
    ]
    
    try:
        while True:
            elapsed = time.time() - start_time
            
            
            for threshold, message in reversed(wait_messages):
                if elapsed >= threshold:
                    status_text.text(message)
                    break
            
            status = api.check_status(snapshot_id)
            
            if status == "ready":
                status_text.text("✅ Data ready! Downloading results...")
                progress_bar.progress(75)
                return api.download_results(snapshot_id)
            elif status == "failed":
                raise Exception("Amazon search failed - please try again")
            
            if elapsed < 30:
                progress = 40 + (elapsed / 30) * 10  # 40-50%
            elif elapsed < 120:
                progress = 50 + ((elapsed - 30) / 90) * 15  # 50-65%
            else:
                progress = 65 + min((elapsed - 120) / 180 * 10, 10)  # 65-75%
            
            progress_bar.progress(int(progress))
            
            time.sleep(5)  # Check every 5 seconds for responsiveness
            
    except Exception as e:
        status_text.error(f"Search failed: {str(e)}")
        raise

if search_clicked and keyword:
    
    progress_container = st.container()
    
    with progress_container:
        try:
            amazon_urls = {
                "US": "https://www.amazon.com",
                "CA": "https://www.amazon.ca",
                "MX": "https://www.amazon.com.mx",
                "GB": "https://www.amazon.co.uk",
                "DE": "https://www.amazon.de",
                "FR": "https://www.amazon.fr",
                "IT": "https://www.amazon.it",
                "ES": "https://www.amazon.es",
                "NL": "https://www.amazon.nl",
                "SE": "https://www.amazon.se",
                "PL": "https://www.amazon.pl",
                "BR": "https://www.amazon.com.br",
                "AU": "https://www.amazon.com.au",
                "JP": "https://www.amazon.co.jp",
                "IN": "https://www.amazon.in",
                "SG": "https://www.amazon.sg",
                "TR": "https://www.amazon.com.tr",
                "SA": "https://www.amazon.sa",
                "AE": "https://www.amazon.ae",
                "ZA": "https://www.amazon.co.za"
            }
            
            amazon_url = amazon_urls.get(countries[selected_country], "https://www.amazon.com")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            time_counter = st.empty()
            skeleton_placeholder = st.empty()
            
            import time
            start_time = time.time()
            
            with skeleton_placeholder.container():
                st.markdown("""
                <div style="animation: pulse 2s infinite; background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%); background-size: 200% 100%; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                    <div style="height: 20px; background: #d0d0d0; border-radius: 4px; margin-bottom: 8px;"></div>
                    <div style="height: 15px; background: #d0d0d0; border-radius: 4px; width: 80%;"></div>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                for i, col in enumerate([col1, col2, col3, col4]):
                    with col:
                        st.markdown("""
                        <div style="animation: pulse 2s infinite; background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%); background-size: 200% 100%; padding: 0.8rem; border-radius: 8px; margin: 0.25rem 0;">
                            <div style="height: 40px; background: #d0d0d0; border-radius: 4px; margin-bottom: 8px;"></div>
                            <div style="height: 12px; background: #d0d0d0; border-radius: 4px; margin-bottom: 4px;"></div>
                            <div style="height: 12px; background: #d0d0d0; border-radius: 4px; width: 60%;"></div>
                        </div>
                        """, unsafe_allow_html=True)
            
            status_text.text("🚀 Initializing search request...")
            progress_bar.progress(10)  # Progress milestone: validation complete
            time.sleep(0.5)  # Brief pause for UX
            
            status_text.text("📡 Connecting to Amazon marketplace...")
            snapshot_id = api.trigger_search(keyword, amazon_url)
            progress_bar.progress(25)  # Progress milestone: API call initiated
            
            results = smart_wait_for_results(api, snapshot_id, progress_bar, status_text, time_counter, start_time)
            
            status_text.text("🔄 Analyzing products and building insights...")
            progress_bar.progress(80)  # Progress milestone: data processing complete
            
            # For debugging and audit
            import json
            import os
            from datetime import datetime
            
            os.makedirs("raw_data", exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_keyword = keyword.replace(' ', '_').replace('/', '_')
            country_code = countries[selected_country]
            raw_filename = f"raw_data/raw_api_{safe_keyword}_{country_code}_{timestamp}.json"
            
            with open(raw_filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            # For debugging and audit
            
            processed_results = processor.process_raw_data(results)
            shopping_intel = intelligence.analyze_products(processed_results)
            
            import uuid
            run_id = str(uuid.uuid4())
            
            st.session_state.search_results = processed_results
            st.session_state.shopping_intelligence = shopping_intel
            st.session_state.current_run_id = run_id
            st.session_state.raw_data = results  # For AI context if needed
            st.session_state.search_metadata = {
                'keyword': keyword,
                'country': countries[selected_country],
                'domain': amazon_url,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            elapsed_time = time.time() - start_time
            status_text.text(f"✅ Found {len(processed_results)} products in {elapsed_time:.1f}s!")
            progress_bar.progress(100)  # Progress milestone: analysis ready
            
            time.sleep(1.5)  # Give user time to see the success message
            progress_bar.empty()
            status_text.empty()
            time_counter.empty()
            skeleton_placeholder.empty()  # Clear skeleton loading screen
            
        except Exception as e:
            try:
                progress_bar.empty()
                status_text.empty()
                time_counter.empty()
                skeleton_placeholder.empty()  # Clear skeleton on error too
            except:
                pass
            
  
            error_msg = str(e)
            
            if hasattr(e, 'last_attempt') and hasattr(e.last_attempt, 'exception'):
                actual_error = e.last_attempt.exception()
                error_msg = str(actual_error) if actual_error else error_msg
                
            if "RetryError" in error_msg or "RetryError" in str(type(e)):
                st.error("⏱️ **Search Timeout** - Bright Data servers are taking longer than expected")
                st.info("💡 **Tip:** Try again in a few moments or try a different search term")
            elif "BrightDataAPIError" in error_msg:
                st.error(f"🔌 **API Error**: {error_msg}")
                st.info("💡 **Tip:** Check your internet connection or try again")
            elif "Timeout" in error_msg:
                st.error("⏱️ **Request Timeout** - The search took too long to complete")
                st.info("💡 **Tip:** Try a more specific search term or try again")
            else:
                st.error(f"❌ **Search Error**: {error_msg}")
                st.info("💡 **Tip:** Try refreshing the page or contact support if the issue persists")
            
            with st.expander("🔧 Technical Details (for debugging)"):
                st.code(str(e))
                import traceback
                st.code(traceback.format_exc())
            
            st.stop()


if st.session_state.search_results:
    results = st.session_state.search_results
    intel = st.session_state.shopping_intelligence
    
    st.markdown("### 📊 Search Results Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Products Found", len(results), delta=None)
    
    with col2:
        prices = [p['final_price'] for p in results if p.get('final_price') is not None]
        avg_price = np.mean(prices) if prices else 0
        current_country_code = countries.get(st.session_state.get('selected_country'), list(countries.values())[0])
        st.metric("💰 Average Price", format_price(avg_price, current_country_code), delta=None)
    
    with col3:
        rated_products = [p['rating'] for p in results if p.get('rating') and p.get('rating') > 0]
        high_rated = len([p for p in results if p.get('rating', 0) >= 4.5])
        if rated_products:
            avg_rating = np.mean(rated_products)
            st.metric("⭐ Rating Quality", f"{avg_rating:.1f}/5 • {high_rated} top rated", delta=None)
        else:
            st.metric("⭐ Rating Quality", "Not rated yet", delta=None)
    
    with col4:
        deal_count = len([p for p in results if 
                         (p.get('discount_pct') is not None and p.get('discount_pct') > 0) or 
                         p.get('is_coupon', False) or
                         'Deal' in str(p.get('badge', ''))
                        ])
        st.metric("🏷️ Active Deals", deal_count, delta=None)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Recommendations", 
        "📊 Market Analysis", 
        "🤖 Product Assistant", 
        "📋 Product Catalog"
    ])
    
    with tab1:
        st.markdown("#### 🏆 Our Top Recommendations for You")
        
        if intel.get('top_picks'):
            picks = intel['top_picks'][:3]
            
            for i, pick in enumerate(picks):
                product = pick['product']
                
                if i > 0:
                    st.markdown("<br>", unsafe_allow_html=True)
                
                with st.expander(f"{pick['reason']} - {product['name'][:60]}...", expanded=True):
                    col1, col2, col3 = st.columns([1.2, 2.5, 1.3])
                    
                    with col1:
                        if product.get('image'):
                            st.image(product['image'], width=120)
                        else:
                            st.markdown("📦<br>No Image", unsafe_allow_html=True)
                    
                    with col2:
                        current_country_code = countries.get(st.session_state.get('selected_country'), list(countries.values())[0])
                        st.write(f"**Price:** {format_price(product['final_price'], current_country_code)}")
                        
                        num_reviews = product.get('num_ratings') or 0
                        if product.get('rating'):
                            st.write(f"**Rating:** {product['rating']:.1f}/5 ({num_reviews:,} reviews)")
                        else:
                            st.write("**Rating:** Not rated yet")
                        
                        if product.get('brand'):
                            st.write(f"**Brand:** {product['brand']}")
                        
                        if product.get('value_score'):
                            st.write(f"**Value Score:** {product['value_score']:.2f}/1.00")
                        
                        deal_badges = []
                        if product.get('is_deal'):
                            deal_badges.append("🏷️ DEAL")
                        if product.get('discount_pct') and product['discount_pct'] > 0:
                            deal_badges.append(f"💰 {product['discount_pct']:.0f}% OFF")
                        if product.get('is_prime'):
                            deal_badges.append("⚡ PRIME")
                        if product.get('is_coupon'):
                            deal_badges.append("🎫 COUPON")
                        
                        if deal_badges:
                            st.markdown("**Deals:** " + " | ".join(deal_badges))
                        
                        st.write(f"**Why this pick:** {pick['reason']}")
                    
                    with col3:
                        st.link_button("🛒 Buy Now", product['url'], help="View and purchase on Amazon", use_container_width=True)
        else:
            st.info("🔍 No recommendations available. Try searching for products first.")
        
    
    with tab2:
        st.markdown("#### 📊 Market Overview")
        
        @st.cache_data
        def compute_market_insights(products_data):
            """Efficiently compute all market insights in a single pass with robust error handling."""
            insights = {
                'prices': [], 'ratings': [], 'value_scores': [], 'brands': [], 'discounts': [],
                'sponsored_prices': [], 'organic_prices': [], 'prime_prices': [], 'non_prime_prices': [],
                'velocity_data': [], 'delivery_data': [], 'positions': [], 'currencies': [], 'total_products': 0, 
                'sponsored_count': 0, 'prime_count': 0, 'deals_count': 0, 'products_with_ratings': 0, 
                'products_with_scores': 0, 'products_with_delivery': 0, 'products_with_position': 0,
                'currency_consistency_issues': 0
            }
            
            if not products_data or not isinstance(products_data, (list, tuple)):
                return insights
            
            for p in products_data:
                if not isinstance(p, dict):
                    continue
                    
                insights['total_products'] += 1
                
                try:
                    final_price = p.get('final_price')
                    currency = p.get('currency', '')
                    
                    if final_price is not None and isinstance(final_price, (int, float)) and final_price >= 0 and not math.isnan(final_price):
                        price = final_price
                        insights['prices'].append(price)
                        
                        if currency and isinstance(currency, str):
                            insights['currencies'].append(currency.upper())
                        
                        if p.get('sponsored'):
                            insights['sponsored_prices'].append(price)
                            insights['sponsored_count'] += 1
                        else:
                            insights['organic_prices'].append(price)
                        
                        if p.get('is_prime'):
                            insights['prime_prices'].append(price)
                            insights['prime_count'] += 1
                        else:
                            insights['non_prime_prices'].append(price)
                except (TypeError, ValueError, AttributeError):
                    pass  # Skip malformed price data
                
                try:
                    rating = p.get('rating')
                    if rating is not None and isinstance(rating, (int, float)) and 0 <= rating <= 5 and not math.isnan(rating):
                        insights['ratings'].append(rating)
                        insights['products_with_ratings'] += 1
                except (TypeError, ValueError, AttributeError):
                    pass  # Skip malformed rating data
                
                try:
                    value_score = p.get('value_score')
                    if value_score is not None and isinstance(value_score, (int, float)) and not math.isnan(value_score):
                        insights['value_scores'].append(value_score)
                        insights['products_with_scores'] += 1
                except (TypeError, ValueError, AttributeError):
                    pass  # Skip malformed value score data
                
                try:
                    brand = p.get('brand')
                    if brand and isinstance(brand, str) and brand.strip():
                        insights['brands'].append(brand.strip())
                except (TypeError, AttributeError):
                    pass  # Skip malformed brand data
                
                try:
                    discount_pct = p.get('discount_pct')
                    if discount_pct is not None and isinstance(discount_pct, (int, float)) and discount_pct > 0 and not math.isnan(discount_pct):
                        insights['discounts'].append(discount_pct)
                except (TypeError, ValueError, AttributeError):
                    pass  # Skip malformed discount data
                
                try:
                    if p.get('is_deal'):
                        insights['deals_count'] += 1
                except (TypeError, AttributeError):
                    pass  # Skip malformed deal data
                
                try:
                    units_past_month = p.get('units_past_month')
                    rating = p.get('rating')
                    if (units_past_month is not None and rating is not None and
                        isinstance(units_past_month, (int, float)) and isinstance(rating, (int, float)) and
                        units_past_month > 0 and 0 <= rating <= 5 and 
                        not math.isnan(units_past_month) and not math.isnan(rating)):
                        insights['velocity_data'].append({
                            'units_sold': units_past_month,
                            'rating': rating,
                            'product_name': str(p.get('name', ''))
                        })
                except (TypeError, ValueError, AttributeError):
                    pass  # Skip malformed velocity data
                
                try:
                    delivery_info = p.get('delivery', [])
                    if isinstance(delivery_info, list) and delivery_info:
                        insights['delivery_data'].extend(delivery_info)
                        insights['products_with_delivery'] += 1
                    elif isinstance(delivery_info, str) and delivery_info.strip():
                        insights['delivery_data'].append(delivery_info.strip())
                        insights['products_with_delivery'] += 1
                except (TypeError, AttributeError):
                    pass  # Skip malformed delivery data
                
                try:
                    position = p.get('position')
                    if position is not None and isinstance(position, (int, float)) and position > 0:
                        insights['positions'].append(int(position))
                        insights['products_with_position'] += 1
                except (TypeError, ValueError, AttributeError):
                    pass  # Skip malformed position data
            
            if insights['currencies']:
                unique_currencies = set(insights['currencies'])
                if len(unique_currencies) > 1:
                    insights['currency_consistency_issues'] = len(unique_currencies) - 1
            
            return insights
        
        market_data = compute_market_insights(results)
        
        
        st.markdown("💰 **Price Distribution**")
        prices = market_data['prices']
        if len(prices) >= 2:  # Need at least 2 data points for meaningful statistics
            current_country_code = countries.get(st.session_state.get('selected_country'), list(countries.values())[0])
            
            q25, q50, q75 = np.percentile(prices, [25, 50, 75])
            iqr = q75 - q25
            mean_price = np.mean(prices)
            
            if iqr > 0:
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                display_prices = [p for p in prices if lower_bound <= p <= upper_bound]
                outlier_count = len(prices) - len(display_prices)
            else:
                display_prices = prices
                outlier_count = 0
            
            if not display_prices:
                display_prices = prices  # Fallback: show all data
                outlier_count = 0
            
            unique_prices = len(set(display_prices))
            nbins = min(20, max(1, unique_prices))  # At least 1 bin
            
            fig_price = px.histogram(
                x=display_prices, 
                nbins=nbins,
                title="💰 Price Range",
                labels={'x': f'Price ({currencies.get(current_country_code, "USD")})', 'y': 'Number of Products'},
                color_discrete_sequence=['#667eea']
            )
            
            if unique_prices > 1:
                fig_price.add_vline(x=q50, line_dash="dash", line_color="orange", annotation_text="Median")
                if iqr > 0:
                    fig_price.add_vline(x=q25, line_dash="dash", line_color="gray", annotation_text="Q1", annotation_position="top")
                    fig_price.add_vline(x=q75, line_dash="dash", line_color="gray", annotation_text="Q3", annotation_position="top")
            
            fig_price.update_layout(
                showlegend=False,
                template="plotly_white",
                title_x=0.5
            )
            st.plotly_chart(fig_price, width="stretch")
            st.caption("📊 Shows how products are distributed across different price points. Helps identify price gaps and opportunities.")
            
        
        st.markdown("⭐ **Quality vs Price Sweet Spot**")
        
        if market_data['prices'] and market_data['ratings']:
            scatter_data = []
            for p in results:
                try:
                    price = p.get('final_price')
                    rating = p.get('rating')
                    num_ratings = p.get('num_ratings', 0)
                    name = p.get('name', '')
                    
                    if (price is not None and isinstance(price, (int, float)) and price >= 0 and not math.isnan(price) and
                        rating is not None and isinstance(rating, (int, float)) and 0 <= rating <= 5 and not math.isnan(rating)):
                        
                        safe_num_ratings = max(1, min(num_ratings or 1, 10000))
                        
                        scatter_data.append({
                            'final_price': price,
                            'rating': rating,
                            'num_ratings': safe_num_ratings,
                            'name': name[:50] + '...' if len(name) > 50 else name  # Truncate long names
                        })
                except (TypeError, ValueError, AttributeError):
                    continue  # Skip malformed data
            
            if scatter_data:
                df_scatter = pd.DataFrame(scatter_data)
                current_country_code = countries.get(st.session_state.get('selected_country'), list(countries.values())[0])
                
                fig_scatter = px.scatter(
                    df_scatter, 
                    x='final_price', 
                    y='rating',
                    size='num_ratings',
                    hover_data=['name', 'num_ratings'],
                    title="⭐ Quality vs Price",
                    labels={'final_price': f'Price ({currencies.get(current_country_code, "USD")})', 'rating': 'Rating (Stars)'},
                    color='rating',
                    color_continuous_scale='Viridis'
                )
                fig_scatter.update_layout(template="plotly_white", title_x=0.5)
                st.plotly_chart(fig_scatter, width="stretch")
                st.caption("🎯 Bubble chart showing the relationship between price and customer ratings. Larger bubbles = more reviews.")
            else:
                st.info("📊 Insufficient valid data for price vs rating analysis")
        else:
            st.info("📊 No pricing or rating data available for analysis")
        
        
        st.markdown("💸 **Deal Analysis**")
        discounts = market_data['discounts']
        if discounts:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_discounts = px.histogram(
                    x=discounts,
                    nbins=15,
                    title="🏷️ Deal Savings",
                    labels={'x': 'Discount %', 'y': 'Number of Products'},
                    color_discrete_sequence=['#ff6b6b']
                )
                fig_discounts.update_layout(template="plotly_white", title_x=0.5)
                st.plotly_chart(fig_discounts, use_container_width=True)
                st.caption("💸 See how many products offer discounts and typical savings amounts.")
            
            with col2:
                avg_discount = np.mean(discounts)
                max_discount = max(discounts)
                deal_products = len(discounts)
                total_products = market_data['total_products']
                
                st.metric("Average Discount", f"{avg_discount:.1f}%")
                st.metric("Max Discount", f"{max_discount:.1f}%")
                st.metric("Products on Sale", f"{deal_products}/{total_products} ({(deal_products/total_products*100):.1f}%)")
        
        
        st.markdown("🎯 **Value Score Distribution**")
        value_scores = market_data['value_scores']
        if value_scores:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_value = px.histogram(
                    x=value_scores,
                    nbins=20,
                    title="🎯 Best Value Products",
                    labels={'x': 'Value Score (0.0-1.0)', 'y': 'Number of Products'},
                    color_discrete_sequence=['#28a745']
                )
                # Add percentile lines for context
                p50 = np.percentile(value_scores, 50)
                p75 = np.percentile(value_scores, 75)
                fig_value.add_vline(x=p50, line_dash="dash", line_color="orange", annotation_text="Median")
                # Value line removed to avoid clutter
                fig_value.update_layout(template="plotly_white", title_x=0.5)
                st.plotly_chart(fig_value, use_container_width=True)
                st.caption("⭐ Our algorithm combines price, quality, and popularity to identify the best overall value products.")
            
            with col2:
                # Value score summary metrics with dynamic thresholds
                avg_value = np.mean(value_scores)
                max_value = max(value_scores)
                
                # FIXED: Accurate top 25% calculation without bias
                if len(value_scores) >= 4:  # Need at least 4 products for meaningful quartile
                    sorted_scores = sorted(value_scores, reverse=True)
                    true_top_25_count = max(1, len(value_scores) // 4)  # Actual 25% count
                    high_value_threshold = sorted_scores[true_top_25_count - 1]
                    
                    # Handle identical values edge case - show actual top 25% count
                    if len(set(value_scores)) == 1:
                        threshold_label = f"={high_value_threshold:.2f}"
                    else:
                        threshold_label = f"≥{high_value_threshold:.2f}"
                    
                    total_with_scores = len(value_scores)
                    percentage = (true_top_25_count / total_with_scores * 100)
                    
                    st.metric("Average Value Score", f"{avg_value:.2f}")
                    st.metric("Best Value Score", f"{max_value:.2f}")
                    st.metric("Top 25% Products", f"{true_top_25_count}/{total_with_scores} ({percentage:.0f}%) {threshold_label}")
                else:
                    # Too few products for quartile analysis
                    st.metric("Average Value Score", f"{avg_value:.2f}")
                    st.metric("Best Value Score", f"{max_value:.2f}")
                    st.metric("Total Scored Products", f"{len(value_scores)}")
        
        st.markdown("---")
        
        # Rating Distribution Analysis - CRITICAL MISSING INSIGHT
        st.markdown("⭐ **Rating Quality Distribution**")
        ratings = market_data['ratings']
        if ratings:
            col1, col2 = st.columns(2)
            
            with col1:
                # Rating distribution histogram
                fig_ratings = px.histogram(
                    x=ratings,
                    nbins=20,
                    title="⭐ Customer Satisfaction",
                    labels={'x': 'Rating (Stars)', 'y': 'Number of Products'},
                    color_discrete_sequence=['#ffd700']
                )
                # Add quality thresholds
                avg_rating = np.mean(ratings)
                fig_ratings.add_vline(x=avg_rating, line_dash="dash", line_color="orange", annotation_text="Average")
                # Rating lines removed to avoid clutter
                # Rating lines removed to avoid clutter
                fig_ratings.update_layout(template="plotly_white", title_x=0.5)
                st.plotly_chart(fig_ratings, use_container_width=True)
                st.caption("😊 Distribution of customer satisfaction ratings across all products found.")
            
            with col2:
                # Rating quality metrics
                excellent_count = len([r for r in ratings if r >= 4.5])
                good_count = len([r for r in ratings if r >= 4.0])
                poor_count = len([r for r in ratings if r < 3.0])
                total_rated = len(ratings)
                
                st.metric("Average Rating", f"{avg_rating:.1f}/5")
                st.metric("Excellent Products (4.5+)", f"{excellent_count}/{total_rated} ({(excellent_count/total_rated*100):.1f}%)")
                st.metric("Poor Products (<3.0)", f"{poor_count}/{total_rated} ({(poor_count/total_rated*100):.1f}%)")
        
        # Search Position Analysis - RANKING INSIGHTS
        st.markdown("📍 **Search Position Analysis**")
        positions = market_data['positions']
        if positions:
            col1, col2 = st.columns(2)
            
            with col1:
                # Position distribution
                position_ranges = []
                for pos in positions:
                    if pos <= 5:
                        position_ranges.append("Top 5")
                    elif pos <= 10:
                        position_ranges.append("6-10")
                    elif pos <= 20:
                        position_ranges.append("11-20")
                    else:
                        position_ranges.append("21+")
                
                if position_ranges:
                    position_counts = pd.Series(position_ranges).value_counts()
                    fig_positions = px.pie(
                        values=position_counts.values,
                        names=position_counts.index,
                        title="📍 Search Rankings",
                        color_discrete_sequence=['#28a745', '#17a2b8', '#ffc107', '#dc3545']
                    )
                    fig_positions.update_layout(template="plotly_white", title_x=0.5)
                    st.plotly_chart(fig_positions, use_container_width=True)
                    st.caption("📍 Shows where products appear in Amazon search results. Lower positions = higher visibility.")
            
            with col2:
                # Position insights
                top_5_count = len([p for p in positions if p <= 5])
                first_page_count = len([p for p in positions if p <= 20])
                avg_position = np.mean(positions)
                
                st.metric("Average Position", f"{avg_position:.1f}")
                st.metric("Top 5 Results", f"{top_5_count}/{len(positions)} ({(top_5_count/len(positions)*100):.1f}%)")
                st.metric("First Page (1-20)", f"{first_page_count}/{len(positions)} ({(first_page_count/len(positions)*100):.1f}%)")
        
        
        # NEW: Competitive Price Position - HIGH BUSINESS VALUE
        st.markdown("🏆 **Competitive Price Position**")
        if market_data['prices'] and len(market_data['prices']) >= 5:
            col1, col2 = st.columns(2)
            
            with col1:
                # Price quartile analysis
                p25, p50, p75 = np.percentile(market_data['prices'], [25, 50, 75])
                current_country_code = countries.get(st.session_state.get('selected_country'), list(countries.values())[0])
                
                # Categorize products by price positioning
                budget_count = len([p for p in market_data['prices'] if p <= p25])
                value_count = len([p for p in market_data['prices'] if p25 < p <= p50])
                premium_count = len([p for p in market_data['prices'] if p50 < p <= p75])
                luxury_count = len([p for p in market_data['prices'] if p > p75])
                
                fig_competition = px.bar(
                    x=[budget_count, value_count, premium_count, luxury_count],
                    y=['Budget\n(Bottom 25%)', 'Value\n(25-50%)', 'Premium\n(50-75%)', 'Luxury\n(Top 25%)'],
                    orientation='h',
                    title="💰 Price Categories",
                    labels={'x': 'Number of Products', 'y': 'Price Tier'},
                    color=[budget_count, value_count, premium_count, luxury_count],
                    color_continuous_scale='RdYlBu_r'
                )
                fig_competition.update_layout(template="plotly_white", title_x=0.5)
                st.plotly_chart(fig_competition, use_container_width=True)
                st.caption("🏷️ Groups products into price tiers to understand market competition and positioning.")
            
            with col2:
                # Price benchmarks
                st.metric("Budget Threshold", format_price(p25, current_country_code))
                st.metric("Market Average", format_price(p50, current_country_code))  
                st.metric("Premium Threshold", format_price(p75, current_country_code))
                
                # Most competitive segment
                max_count = max(budget_count, value_count, premium_count, luxury_count)
                if budget_count == max_count:
                    st.metric("Most Competition", "Budget segment")
                elif value_count == max_count:
                    st.metric("Most Competition", "Value segment")
                elif premium_count == max_count:
                    st.metric("Most Competition", "Premium segment")
                else:
                    st.metric("Most Competition", "Luxury segment")
        elif len(market_data['prices']) > 0:
            st.metric("Products Available", len(market_data['prices']))
    
    with tab3:
        st.markdown("#### 🤖 Product Assistant")
        
        # Check if we have fresh search data
        current_run_id = st.session_state.get('current_run_id')
        if not current_run_id:
            st.info("🔍 Perform a search to ask questions about your results")
        
        # Initialize AI query in session state if not exists
        if "ai_input" not in st.session_state:
            st.session_state.ai_input = ""
        
        
        # AI query input with practical examples
        ai_query = st.text_input(
            "💬 Ask me anything about these products:",
            placeholder="e.g., 'Which product has the best value?' or 'Show me the top-rated options'",
            key="ai_input", 
            disabled=not current_run_id  # Disable if no search data
        )
        
        if st.button("🤖 Ask Assistant", type="primary", disabled=not current_run_id, help="Ask questions about your search results") and ai_query:
            with st.spinner("Analyzing data..."):
                # Query the Gemini AI engine
                # AI query processing for enhanced product analysis
                # Pass current session data directly to AI engine
                ai_response = ai_engine.query_with_data(
                    question=ai_query,
                    products=st.session_state.search_results,
                    metadata=st.session_state.get('search_metadata', {})
                )
                
                if ai_response.success:
                    st.markdown("**Results:**")
                    st.markdown(ai_response.response)
                    # Clean response - no technical details
                    
                else:
                    st.error("❌ Analysis failed")
                    st.error(ai_response.response)
        
    
    with tab4:
        st.markdown("#### 📋 Product Results")
        st.markdown(f"🎯 **{market_data['total_products']} products found**")
        
        # Initialize filter states
        if 'catalog_sort_by' not in st.session_state:
            st.session_state.catalog_sort_by = None
        if 'catalog_filter' not in st.session_state:
            st.session_state.catalog_filter = 'all'
        if 'catalog_search' not in st.session_state:
            st.session_state.catalog_search = ''
        
        # Analyze all products to determine which fields have data - optimized for user value
        all_fields = {
            'name': 'Product Name',
            'final_price': 'Price',
            'initial_price': 'Original Price', 
            'discount_pct': 'Discount %',
            'rating': 'Rating',
            'num_ratings': 'Reviews',
            'value_score': 'Value Score',
            'units_past_month': 'Units Sold/Month',
            'position': 'Search Rank',
            'badges': 'Badges',
            'is_deal': 'Deal',
            'is_prime': 'Prime',
            'is_coupon': 'Coupon',
            'brand': 'Brand',
            'delivery': 'Shipping'
        }
        
        # Find fields that have at least one non-empty value
        active_fields = {}
        for field, display_name in all_fields.items():
            has_data = any(
                product.get(field) is not None and 
                str(product.get(field)).strip() != '' and 
                product.get(field) != 'N/A' or 
                field in ['final_price', 'rating', 'units_past_month', 'position', 'value_score']  # Always show these even if 0
                for product in results
            )
            if has_data:
                active_fields[field] = display_name
        
        current_country_code = countries.get(st.session_state.get('selected_country'), list(countries.values())[0])
        df_data = []
        
        for i, product in enumerate(results, 1):
            row_data = {'#': i}
            
            for field, display_name in active_fields.items():
                value = product.get(field)
                
                # Format each field type appropriately
                if field == 'name':
                    row_data[display_name] = str(value or 'N/A')[:60] + ('...' if len(str(value or '')) > 60 else '')  # 60 chars prevent table overflow
                elif field == 'final_price':
                    row_data[display_name] = format_price(float(value or 0), current_country_code)
                elif field == 'initial_price':
                    row_data[display_name] = format_price(float(value or 0), current_country_code)
                elif field == 'discount_pct':
                    row_data[display_name] = f"{float(value or 0):.1f}%" if value else "0%"
                elif field == 'rating':
                    row_data[display_name] = f"{float(value or 0):.1f}/5"
                elif field == 'num_ratings':
                    row_data[display_name] = f"{int(value or 0):,}"
                elif field == 'value_score':
                    row_data[display_name] = f"{float(value or 0):.2f}" if value else "0.00"
                elif field == 'units_past_month':
                    row_data[display_name] = f"{int(value or 0):,}"
                elif field == 'position':
                    row_data[display_name] = f"#{int(value or 0)}" if value else "N/A"
                elif field == 'badges':
                    if isinstance(value, list) and value:
                        badges_text = ', '.join(value[:2])  # Show max 2 badges
                        row_data[display_name] = badges_text[:30] + ('...' if len(badges_text) > 30 else '')
                    else:
                        row_data[display_name] = str(value) if value else "None"
                elif field == 'is_deal':
                    row_data[display_name] = "🔥 Yes" if value else "No"
                elif field in ['is_prime', 'is_coupon']:
                    row_data[display_name] = "✅ Yes" if value else "No"
                elif field == 'delivery':
                    if isinstance(value, list) and value:
                        # Show first delivery option, truncated
                        delivery_text = value[0]
                        row_data[display_name] = delivery_text[:25] + ('...' if len(delivery_text) > 25 else '')
                    else:
                        row_data[display_name] = str(value) if value else "N/A"
                else:
                    row_data[display_name] = str(value or 'N/A')
            
            df_data.append(row_data)
        
        df = pd.DataFrame(df_data)
        
        # Filter controls - Better aligned layout
        col1, col2, col3, col_reset = st.columns([2.5, 2.5, 3, 2])
        
        with col1:
            sort_option = st.selectbox(
                "🔄 Sort by:",
                ["Best Match", "Best Value", "Price: Low to High", "Price: High to Low", "Top Rated", "Most Reviews", "Most Popular"],
                key="catalog_sort_select"
            )
        
        with col2:
            filter_option = st.selectbox(
                "🎯 Filter:",
                ["All Products", "Prime Only", "Best Value (0.8+)", "Highly Rated (4.5+)", "With Discounts"],
                key="catalog_filter_select"
            )
        
        with col3:
            # Dynamic price ranges based on selected market
            selected_country = st.session_state.get('selected_country')
            current_country_code = countries.get(selected_country, list(countries.values())[0])
            currency_code = currencies.get(current_country_code, "USD")
            symbol = currency_symbols.get(currency_code, "$")
            
            # Adjust price ranges for different currencies
            if currency_code == "JPY":
                price_ranges = ["All Prices", f"{symbol}0-2,500", f"{symbol}2,500-5,000", f"{symbol}5,000-10,000", f"{symbol}10,000-20,000", f"{symbol}20,000+"]
                range_values = [(0, 2500), (2500, 5000), (5000, 10000), (10000, 20000), (20000, float('inf'))]
            elif currency_code in ["EUR", "GBP"]:
                price_ranges = ["All Prices", f"{symbol}0-20", f"{symbol}20-40", f"{symbol}40-80", f"{symbol}80-160", f"{symbol}160+"]
                range_values = [(0, 20), (20, 40), (40, 80), (80, 160), (160, float('inf'))]
            elif currency_code == "INR":
                price_ranges = ["All Prices", f"{symbol}0-2,000", f"{symbol}2,000-4,000", f"{symbol}4,000-8,000", f"{symbol}8,000-16,000", f"{symbol}16,000+"]
                range_values = [(0, 2000), (2000, 4000), (4000, 8000), (8000, 16000), (16000, float('inf'))]
            else:  # USD and similar currencies
                price_ranges = ["All Prices", f"{symbol}0-25", f"{symbol}25-50", f"{symbol}50-100", f"{symbol}100-200", f"{symbol}200+"]
                range_values = [(0, 25), (25, 50), (50, 100), (100, 200), (200, float('inf'))]
            
            price_range = st.selectbox(
                "💰 Price Range:",
                price_ranges,
                key="catalog_price_range"
            )
        
        with col_reset:
            # Properly aligned reset button with vertical padding
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            if st.button("🔄 Reset", use_container_width=True, help="Clear all filters and search"):
                # Remove widget keys to reset to defaults
                keys_to_reset = ['catalog_sort_select', 'catalog_filter_select', 'catalog_price_range', 'catalog_search_input']
                for key in keys_to_reset:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        
        # Advanced filters row  
        search_term = st.text_input("🔍 Quick Search:", placeholder="Product name, ASIN, or keyword...", key="catalog_search_input")
        
        # Apply filters to the DataFrame
        filtered_df = df.copy()
        
        # Apply text search
        if search_term:
            mask = pd.Series([False] * len(filtered_df))
            if 'Product Name' in filtered_df.columns:
                mask = mask | filtered_df['Product Name'].str.contains(search_term, case=False, na=False)
            if 'ASIN' in filtered_df.columns:
                mask = mask | filtered_df['ASIN'].str.contains(search_term, case=False, na=False)
            if 'Brand' in filtered_df.columns:
                mask = mask | filtered_df['Brand'].str.contains(search_term, case=False, na=False)
            filtered_df = filtered_df[mask]
        
        # Apply category filters with robust error handling
        try:
            if filter_option == "Prime Only" and 'Prime' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Prime'] == "✅ Yes"]
            elif filter_option == "Best Value (0.8+)" and 'Value Score' in filtered_df.columns:
                value_col = pd.to_numeric(filtered_df['Value Score'], errors='coerce').fillna(0)
                filtered_df = filtered_df[value_col >= 0.8]
            elif filter_option == "Highly Rated (4.5+)" and 'Rating' in filtered_df.columns:
                rating_col = pd.to_numeric(filtered_df['Rating'].str.extract(r'(\d+\.\d+)')[0], errors='coerce').fillna(0)
                filtered_df = filtered_df[rating_col >= 4.5]
            elif filter_option == "With Discounts" and 'Discount %' in filtered_df.columns:
                discount_col = pd.to_numeric(filtered_df['Discount %'].str.replace('%', ''), errors='coerce').fillna(0)
                filtered_df = filtered_df[discount_col > 0]
        except Exception:
            pass  # Keep original data if filtering fails
        
        # Apply price range filter with dynamic currency support
        try:
            if price_range != "All Prices" and 'Price' in filtered_df.columns:
                price_col = pd.to_numeric(filtered_df['Price'].str.replace(r'[^\d.,]', '', regex=True).str.replace(',', ''), errors='coerce').fillna(0)
                
                # Find which range was selected
                selected_range_index = price_ranges.index(price_range) - 1  # -1 because "All Prices" is index 0
                if 0 <= selected_range_index < len(range_values):
                    min_price, max_price = range_values[selected_range_index]
                    if max_price == float('inf'):
                        filtered_df = filtered_df[price_col > min_price]
                    else:
                        filtered_df = filtered_df[(price_col > min_price) & (price_col <= max_price)]
        except Exception:
            pass  # Keep original data if price filtering fails
        
        # Apply sorting with robust error handling
        try:
            if sort_option == "Best Match":
                # Keep original order (default behavior)
                pass
            elif sort_option == "Price: Low to High" and 'Price' in filtered_df.columns:
                price_col = pd.to_numeric(filtered_df['Price'].str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0)
                filtered_df = filtered_df.iloc[price_col.argsort()]
            elif sort_option == "Price: High to Low" and 'Price' in filtered_df.columns:
                price_col = pd.to_numeric(filtered_df['Price'].str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0)
                filtered_df = filtered_df.iloc[price_col.argsort()[::-1]]
            elif sort_option == "Top Rated" and 'Rating' in filtered_df.columns:
                rating_col = pd.to_numeric(filtered_df['Rating'].str.extract(r'(\d+\.\d+)')[0], errors='coerce').fillna(0)
                filtered_df = filtered_df.iloc[rating_col.argsort()[::-1]]
            elif sort_option == "Most Reviews" and 'Reviews' in filtered_df.columns:
                reviews_col = pd.to_numeric(filtered_df['Reviews'].str.replace(r'[^\d]', '', regex=True), errors='coerce').fillna(0)
                filtered_df = filtered_df.iloc[reviews_col.argsort()[::-1]]
            elif sort_option == "Best Value" and 'Value Score' in filtered_df.columns:
                value_col = pd.to_numeric(filtered_df['Value Score'], errors='coerce').fillna(0)
                filtered_df = filtered_df.iloc[value_col.argsort()[::-1]]
            elif sort_option == "Most Popular":
                # Check which column exists and use it
                if 'Units Sold' in filtered_df.columns:
                    units_col = pd.to_numeric(filtered_df['Units Sold'].str.replace(r'[^\d]', '', regex=True), errors='coerce').fillna(0)
                    filtered_df = filtered_df.iloc[units_col.argsort()[::-1]]
                elif 'Bought Last Month' in filtered_df.columns:
                    units_col = pd.to_numeric(filtered_df['Bought Last Month'].str.replace(r'[^\d]', '', regex=True), errors='coerce').fillna(0)
                    filtered_df = filtered_df.iloc[units_col.argsort()[::-1]]
        except Exception:
            # If sorting fails, keep original order
            pass
        
        # Configure responsive column widths that adapt to content
        column_config = {'#': st.column_config.NumberColumn("#", width="small")}
        
        # Calculate dynamic widths based on content and column names
        num_columns = len(active_fields) + 1  # +1 for index column
        
        for field, display_name in active_fields.items():
            # Analyze content to determine optimal width
            if field == 'name':
                # Product names need more space - use large width
                column_config[display_name] = st.column_config.TextColumn(display_name, width="large")
            elif field in ['final_price', 'initial_price']:
                # Price fields - medium width for currency formatting
                column_config[display_name] = st.column_config.TextColumn(display_name, width="medium")
            elif field in ['discount_pct', 'rating', 'value_score']:
                # Numeric fields - small width
                column_config[display_name] = st.column_config.TextColumn(display_name, width="small")
            elif field == 'num_ratings':
                # Review count - small width
                column_config[display_name] = st.column_config.TextColumn(display_name, width="small") 
            elif field == 'brand':
                # Brand names - medium width
                column_config[display_name] = st.column_config.TextColumn(display_name, width="medium")
            elif field == 'url':
                # Link column - small width with clear action text
                column_config[display_name] = st.column_config.LinkColumn(display_name, width="small", display_text="🛒 Buy")
            elif field in ['is_prime', 'is_coupon', 'is_subscribe_and_save', 'sponsored']:
                # Boolean fields - small width
                column_config[display_name] = st.column_config.TextColumn(display_name, width="small")
            elif len(display_name) > 15:
                # Long column names get medium width
                column_config[display_name] = st.column_config.TextColumn(display_name, width="medium")
            else:
                # Default to small for efficiency
                column_config[display_name] = st.column_config.TextColumn(display_name, width="small")
        
        # Display filtered results info
        if len(filtered_df) != len(df):
            st.info(f"🎯 Showing {len(filtered_df)} of {len(df)} products matching your filters")
        
        # Display the filtered dataframe
        st.dataframe(
            filtered_df,
            width="stretch",
            hide_index=True,
            column_config=column_config,
            height=600
        )
        
        # Download functionality
        if len(filtered_df) > 0:
            csv = filtered_df.to_csv(index=False)
            current_datetime = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"amazon_products_{current_datetime}.csv"
            
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=filename,
                mime="text/csv",
                type="primary",
                use_container_width=True
            )

else:
    # Clean welcome screen
    st.markdown("""
    <div style="text-align: center; padding: 4rem 0;">
    </div>
    """, unsafe_allow_html=True)

