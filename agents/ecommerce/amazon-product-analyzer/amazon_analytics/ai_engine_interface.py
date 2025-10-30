"""Abstract interface for AI engines ensuring consistent behavior across providers."""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from .ai_response import StandardAIResponse


class AIEngineInterface(ABC):
    """Abstract base for AI engines with standardized query processing and monitoring."""
    
    def __init__(self):
        """Initialize engine with class name identification."""
        self.name = self.__class__.__name__
    
    @abstractmethod
    def query(self, question: str, run_id: Optional[str] = None) -> StandardAIResponse:
        """Process query and return standardized AI response."""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Return engine health metrics and operational status."""
        pass
    
    @abstractmethod
    def validate_data_consistency(self, run_id: Optional[str] = None) -> Tuple[bool, List[str]]:
        """Verify data integrity and consistency."""
        pass
    
    @abstractmethod  
    def get_capabilities(self) -> List[str]:
        """Return list of supported engine capabilities."""
        pass
    
    def pre_query_validation(self, question: str, run_id: Optional[str] = None) -> Tuple[bool, str]:
        """Validate query parameters before processing."""
        if not question or not question.strip():
            return False, "Question cannot be empty"
        
        if len(question) > 10000:  # Reasonable length limit
            return False, "Question is too long (max 10000 characters)"
        
        return True, ""
    
    def post_response_validation(self, response: StandardAIResponse) -> StandardAIResponse:
        """Validate and enhance response before returning."""
        if not response.query:
            response.add_warning("Missing query in response")
        
        if not response.response:
            response.add_warning("Empty response generated")
        
        if response.confidence < 0.3:
            response.add_warning("Low confidence response")
        
        return response
    
    def should_use_fallback(self, question: str, context: Dict[str, Any]) -> bool:
        """Determine if query should use fallback processing."""
        return False
    
    def create_fallback_response(self, question: str, reason: str) -> StandardAIResponse:
        """Create fallback response when primary processing fails."""
        from datetime import datetime
        
        return StandardAIResponse(
            success=False,
            query=question,
            response=f"Unable to process query: {reason}. Please try rephrasing your question or try again later.",
            confidence=0.0,
            execution_time=0.0,
            timestamp=datetime.utcnow(),
            products_analyzed=0,
            data_source="fallback",
            verification_passed=False,
            fact_check_score=0.0,
            computed_facts={},
            reasoning_chain=[f"Fallback triggered: {reason}"],
            analysis_method="fallback",
            metadata={"fallback_reason": reason},
            warnings=[f"Fallback response due to: {reason}"]
        )
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get engine implementation details for monitoring."""
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'module': self.__class__.__module__,
            'abstract_methods': [method for method in dir(self) 
                               if getattr(getattr(self, method), '__isabstractmethod__', False)],
            'capabilities': self.get_capabilities() if hasattr(self, 'get_capabilities') else [],
            'description': self.__doc__ or "No description available"
        }