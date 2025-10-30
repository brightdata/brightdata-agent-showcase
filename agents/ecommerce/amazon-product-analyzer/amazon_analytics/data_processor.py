"""Data processing pipeline using Pandas for cleaning and feature engineering."""
import pandas as pd
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid
import math

class DataProcessor:
    """Handles data cleaning, normalization, and feature engineering."""
    
    def __init__(self):
        self.currency_symbols = {
            "€": "EUR", "£": "GBP", "$": "USD", "¥": "JPY", "₹": "INR",
            "₽": "RUB", "¢": "USD", "₨": "INR", "₩": "KRW", "₦": "NGN",
            "₡": "CRC", "₴": "UAH", "₵": "GHS", "₸": "KZT", "₼": "AZN"
        }
        
        # Order matters: more specific domains first to prevent partial matches
        self.domain_to_market = {
            "amazon.com.mx": "MX", "amazon.com.br": "BR", "amazon.com.be": "BE", 
            "amazon.com.tr": "TR", "amazon.com.au": "AU", "amazon.co.uk": "UK", 
            "amazon.co.jp": "JP", "amazon.co.za": "ZA", "amazon.com": "US", 
            "amazon.ca": "CA", "amazon.ie": "IE", "amazon.de": "DE", 
            "amazon.fr": "FR", "amazon.it": "IT", "amazon.es": "ES",
            "amazon.nl": "NL", "amazon.se": "SE", "amazon.pl": "PL",
            "amazon.ae": "AE", "amazon.sa": "SA", "amazon.eg": "EG",
            "amazon.in": "IN", "amazon.sg": "SG"
        }
    
    def extract_market_from_domain(self, domain: str) -> str:
        """Extract market code from Amazon domain.
        
        Args:
            domain (str): Amazon domain URL (e.g., 'amazon.com', 'amazon.co.uk')
            
        Returns:
            str: Two-letter market code (e.g., 'US', 'UK') or 'UNKNOWN'
        """
        for d, market in self.domain_to_market.items():
            if d in domain:
                return market
        return "UNKNOWN"

    def parse_float_locale(self, value: Any) -> Optional[float]:
        """Robust float parser handling international number formats.
        
        Handles European (1.234,56) and American (1,234.56) formats by analyzing
        comma/period patterns to determine which is decimal vs thousands separator.
        """
        if value is None or value == "":
            return None
        if isinstance(value, (int, float)):
            try:
                return float(value)
            except Exception:
                return None
        if isinstance(value, str):
            s = re.sub(r"[^0-9.,]", "", value)
            if not s:
                return None
            has_comma = "," in s
            has_dot = "." in s
            try:
                if has_comma and has_dot:
                    # Decide decimal by last separator
                    last_comma = s.rfind(',')
                    last_dot = s.rfind('.')
                    if last_comma > last_dot:
                        s = s.replace('.', '')
                        s = s.replace(',', '.')
                    else:
                        s = s.replace(',', '')
                elif has_comma and not has_dot:
                    # If comma then 3 digits: thousands
                    if re.search(r",\d{3}$", s):
                        s = s.replace(',', '')
                    else:
                        s = s.replace(',', '.')
                else:
                    # Only dot or digits
                    if re.search(r"\.\d{3}$", s):
                        s = s.replace('.', '')
                return float(s)
            except Exception:
                return None
        return None
    
    
    def clean_boolean(self, value: Any) -> bool:
        """Convert various boolean representations to bool."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        if isinstance(value, (int, float)):
            return bool(value)
        return False
    
    def clean_integer(self, value: Any) -> Optional[int]:
        """Clean and convert to integer."""
        if value is None or value == "":
            return None
        
        if isinstance(value, int):
            return value
        
        if isinstance(value, float):
            return int(value) if not math.isnan(value) else None
        
        if isinstance(value, str):
            clean_value = re.sub(r'[^\d]', '', value)
            try:
                return int(clean_value) if clean_value else None
            except ValueError:
                return None
        
        return None
    
    def compute_discount_percentage(self, initial_price: Optional[float], final_price: Optional[float]) -> Optional[float]:
        """Compute discount percentage, only if initial price is valid."""
        if not initial_price or not final_price or initial_price <= 0 or final_price <= 0:
            return None
        
        if initial_price <= final_price:
            return 0.0
        
        return round(((initial_price - final_price) / initial_price) * 100, 2)
    
    def is_deal(self, badges: List[str]) -> bool:
        """Check if product has deal badges."""
        if not badges:
            return False
            
        deal_indicators = [
            'deal', 'sale', 'discount', 'limited time', 'offer', 'special',
            'befristetes angebot', 'angebot', 'rabatt', 'aktion',
            'offre', 'promo', 'reduction', 'descuento', 'oferta',
            'sconto', 'promozione', 'korting', 'aanbieding'
        ]
        
        badges_text = ' '.join(badges).lower()
        return any(indicator in badges_text for indicator in deal_indicators)
    
    def compute_value_score(
        self, 
        rating: Optional[float], 
        num_ratings: Optional[int], 
        discount_pct: Optional[float],
        min_reviews: int = 10
    ) -> float:
        """Compute a value score for ranking products.
        
        Business Logic:
        - 40% weight on quality (rating) - most important factor for customer satisfaction
        - 30% weight on social proof (review volume) - logarithmic scale to prevent gaming
        - 30% weight on deal value (discount) - capped at 50% to maintain quality focus
        
        Returns:
            float: Composite score 0.0-1.0, higher indicates better overall value
        """
        score = 0.0
        
        # Quality component (40%)
        if rating and rating > 0:
            score += (rating / 5.0) * 0.4  # Normalize to 0.4 max (40% weight)
        
        # Social proof component (30%)
        if num_ratings and num_ratings >= min_reviews:
            # Log scale prevents review manipulation
            review_score = min(math.log10(num_ratings) / 4, 1.0)
            score += review_score * 0.3
        
        # Deal component (30%)
        if discount_pct and discount_pct > 0:
            discount_score = min(discount_pct / 50, 1.0)
            score += discount_score * 0.3
        
        return round(score, 2)
    
    def process_raw_data(self, raw_data: List[Dict[str, Any]], search_run_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Process raw API data into cleaned, normalized format."""
        if not raw_data:
            return []
        
        if search_run_id is None:
            search_run_id = str(uuid.uuid4())
        
        processed_items = []
        
        for item in raw_data:
            try:
                domain = item.get('domain', '')
                market = self.extract_market_from_domain(domain)
                
                initial_price = self.parse_float_locale(item.get('initial_price'))
                final_price = self.parse_float_locale(item.get('final_price'))
                
                rating = self.parse_float_locale(item.get('rating'))
                if rating and (rating < 0 or rating > 5):
                    rating = None
                
                num_ratings = self.clean_integer(item.get('num_ratings'))
                
                sponsored = self.clean_boolean(item.get('sponsored'))
                
                discount_pct = self.compute_discount_percentage(initial_price, final_price)
                
                badges = item.get('badge') if 'badge' in item else item.get('badges', [])
                if isinstance(badges, str):
                    badges = [badges] if badges else []
                elif not isinstance(badges, list):
                    badges = []
                
                is_deal_flag = self.is_deal(badges)
                
                units_past_month = (
                    self.clean_integer(item.get('sold')) or 
                    self.clean_integer(item.get('bought_past_month')) or 
                    None
                )
                
                position = None
                page_number = self.clean_integer(item.get('page_number'))
                rank_on_page = self.clean_integer(item.get('rank_on_page'))
                if page_number and rank_on_page:
                    position = (page_number - 1) * 20 + rank_on_page
                
                delivery_info = item.get('delivery', [])
                if isinstance(delivery_info, str):
                    delivery_info = [delivery_info]
                elif not isinstance(delivery_info, list):
                    delivery_info = []
                
                value_score = self.compute_value_score(rating, num_ratings, discount_pct)
                
                raw_brand = item.get('brand', '')
                brand = None
                if isinstance(raw_brand, str):
                    rb = raw_brand.strip()
                    brand = rb if rb else None

                processed_item = {
                    'search_run_id': search_run_id,
                    'asin': item.get('asin', ''),
                    'url': item.get('url', ''),
                    'name': item.get('name', '').strip(),
                    'sponsored': sponsored,
                    'initial_price': initial_price,
                    'final_price': final_price,
                    'currency': item.get('currency', ''),
                    'rating': rating,
                    'num_ratings': num_ratings,
                    'discount_pct': discount_pct,
                    'badges': badges,
                    'is_deal': is_deal_flag,
                    'units_past_month': units_past_month,
                    'position': position,
                    'page_number': page_number,
                    'rank_on_page': rank_on_page,
                    'market': market,
                    'domain': domain,
                    'brand': brand,
                    'image': item.get('image', ''),
                    'delivery': delivery_info,
                    'is_prime': self.clean_boolean(item.get('is_prime')),
                    'is_subscribe_and_save': self.clean_boolean(item.get('is_subscribe_and_save')),
                    'is_coupon': self.clean_boolean(item.get('is_coupon')),
                    'value_score': value_score,
                    'keyword': item.get('keyword', ''),
                    'timestamp': item.get('timestamp', datetime.utcnow().isoformat()),
                    'variations': item.get('variations', [])
                }
                
                processed_items.append(processed_item)
                
            except Exception as e:
                continue
        
        return self.deduplicate_items(processed_items)
    
    def deduplicate_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate items by (search_run_id, asin), keeping best ranked."""
        if not items:
            return items
        
        try:
            normalized_items = []
            for item in items:
                normalized_item = item.copy()
                
                if 'variations' in normalized_item:
                    variations = normalized_item['variations']
                    if isinstance(variations, list):
                        normalized_item['variations'] = variations
                    elif not variations:
                        normalized_item['variations'] = []
                    else:
                        normalized_item['variations'] = [str(variations)]
                
                if 'badges' in normalized_item:
                    badges = normalized_item['badges']
                    if isinstance(badges, list):
                        normalized_item['badges'] = badges
                    else:
                        normalized_item['badges'] = [str(badges)] if badges else []
                
                if 'delivery' in normalized_item:
                    delivery = normalized_item['delivery']
                    if isinstance(delivery, list):
                        normalized_item['delivery'] = delivery
                    else:
                        normalized_item['delivery'] = [str(delivery)] if delivery else []
                
                normalized_items.append(normalized_item)
            
            df = pd.DataFrame(normalized_items)
            
            if df.empty:
                return items
            
            # Deduplicate by search_run_id and asin, keeping the first (best ranked)
            deduped = (
                df
                .sort_values(['search_run_id', 'asin', 'position'])
                .groupby(['search_run_id', 'asin'])
                .first()
                .reset_index()
            )
            
            return deduped.to_dict('records')
            
        except Exception as e:
            
            seen = set()
            deduped_items = []
            
            # Sort by position to keep best ranked items
            sorted_items = sorted(items, key=lambda x: (x.get('search_run_id', ''), x.get('asin', ''), x.get('position', 999)))
            
            for item in sorted_items:
                key = (item.get('search_run_id', ''), item.get('asin', ''))
                if key not in seen:
                    seen.add(key)
                    deduped_items.append(item)
            
            return deduped_items
    
