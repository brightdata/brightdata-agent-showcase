"""Shopping Intelligence Engine - Product Recommendation System"""
from typing import Dict, List, Any


class ShoppingIntelligenceEngine:
    """Generate product recommendations from shopping data."""
    
    def __init__(self):
        pass
    
    def analyze_products(self, products: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate shopping intelligence from product data."""
        if not products:
            return {'total_items': 0, 'top_picks': []}
        
        top_picks = self._generate_top_picks(products)
        
        return {
            'total_items': len(products),
            'top_picks': top_picks
        }
    
    def _generate_top_picks(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate top product recommendations with reasoning."""
        try:
            valid_products = []
            for product in products:
                rating = product.get('rating')
                price = product.get('final_price')
                num_ratings = product.get('num_ratings', 0)
                
                if rating is not None and price is not None and rating > 0 and price > 0:
                    valid_products.append(product)
            
            if not valid_products:
                return []
            
            picks = []
            used_asins = set()
            
            best_value = self._find_best_value(valid_products)
            if best_value and best_value.get('asin') not in used_asins:
                picks.append({
                    'product': best_value,
                    'reason': 'Best Overall Value',
                    'explanation': 'Excellent balance of quality, price, and customer reviews'
                })
                used_asins.add(best_value['asin'])
            
            highest_rated = self._find_highest_rated(valid_products)
            if highest_rated and highest_rated.get('asin') not in used_asins:
                picks.append({
                    'product': highest_rated,
                    'reason': 'Highest Rated',
                    'explanation': 'Top customer satisfaction with proven track record'
                })
                used_asins.add(highest_rated['asin'])
            
            best_deal = self._find_best_deal(valid_products)
            if best_deal and best_deal.get('asin') not in used_asins:
                picks.append({
                    'product': best_deal,
                    'reason': 'Best Deal',
                    'explanation': 'Great value with significant savings and good quality'
                })
                used_asins.add(best_deal['asin'])
            
            if len(picks) < 3:
                remaining_products = [p for p in valid_products if p.get('asin') not in used_asins]
                remaining_products.sort(key=lambda x: x.get('value_score', 0), reverse=True)
                
                for product in remaining_products[:3-len(picks)]:
                    picks.append({
                        'product': product,
                        'reason': 'Quality Choice',
                        'explanation': 'Good balance of quality and value'
                    })
            
            return picks[:3]
            
        except Exception:
            return []
    
    def _find_best_value(self, products: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find product with best value score."""
        candidates = [p for p in products if 
                     p.get('value_score') is not None and 
                     p.get('num_ratings', 0) >= 10]
        
        if not candidates:
            return None
        
        return max(candidates, key=lambda p: p.get('value_score', 0))
    
    def _find_highest_rated(self, products: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find highest rated product with sufficient reviews."""
        candidates = [p for p in products if 
                     p.get('rating', 0) >= 4.0 and 
                     p.get('num_ratings', 0) >= 50]
        
        if not candidates:
            return None
        
        return max(candidates, key=lambda p: (p.get('rating', 0), p.get('num_ratings', 0)))
    
    def _find_best_deal(self, products: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find product with best discount that maintains quality."""
        candidates = [p for p in products if 
                     p.get('discount_pct') is not None and 
                     p.get('discount_pct', 0) >= 10 and
                     p.get('rating', 0) >= 3.5]
        
        if not candidates:
            return None
        
        return max(candidates, key=lambda p: p.get('discount_pct', 0))