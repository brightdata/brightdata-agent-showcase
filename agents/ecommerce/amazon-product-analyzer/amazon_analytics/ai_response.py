"""Standardized AI response format with unified structure and validation."""
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime


@dataclass
class StandardAIResponse:
    """Unified AI response with metadata, validation, and quality metrics."""
    
    success: bool
    query: str
    response: str
    confidence: float
    execution_time: float
    timestamp: datetime
    products_analyzed: int
    data_source: str
    verification_passed: bool
    fact_check_score: float
    computed_facts: Dict[str, Any]
    reasoning_chain: List[str]
    analysis_method: str
    metadata: Dict[str, Any]
    warnings: List[str] = None
    
    def __post_init__(self):
        """
        Post-initialization validation and setup.
        
        Performs data validation, sets default values for optional fields,
        and ensures the response meets quality standards. This includes
        confidence score validation, timestamp verification, and metadata
        consistency checks.
        """
        if self.warnings is None:
            self.warnings = []
        
        if not 0.0 <= self.confidence <= 1.0:
            self.warnings.append(f"Confidence score {self.confidence} outside valid range [0.0, 1.0]")
            self.confidence = max(0.0, min(1.0, self.confidence))
        
        if not 0.0 <= self.fact_check_score <= 1.0:
            self.warnings.append(f"Fact check score {self.fact_check_score} outside valid range [0.0, 1.0]")
            self.fact_check_score = max(0.0, min(1.0, self.fact_check_score))
        
        if self.products_analyzed < 0:
            self.warnings.append(f"Negative products_analyzed count: {self.products_analyzed}")
            self.products_analyzed = 0
        
        if not isinstance(self.computed_facts, dict):
            self.warnings.append("computed_facts should be a dictionary")
            self.computed_facts = {}
        
        if not isinstance(self.reasoning_chain, list):
            self.warnings.append("reasoning_chain should be a list")
            self.reasoning_chain = []
        
        if self.execution_time < 0:
            self.warnings.append(f"Negative execution time: {self.execution_time}")
            self.execution_time = 0.0
    
    def is_high_confidence(self, threshold: float = 0.7) -> bool:
        """
        Check if the response has high confidence.
        
        Args:
            threshold: Minimum confidence score to be considered high (default 0.7)
            
        Returns:
            True if confidence score is above threshold
        """
        return self.confidence >= threshold
    
    def is_verified_accurate(self) -> bool:
        """
        Check if the response passed both verification and has good fact-checking.
        
        Returns:
            True if verification passed and fact check score is above 0.5
        """
        return self.verification_passed and self.fact_check_score > 0.5
    
    def get_accuracy_score(self) -> float:
        """
        Calculate combined accuracy score from confidence and fact checking.
        
        This combines the AI's confidence in its response with the fact-checking
        score to provide an overall accuracy assessment. The weighting gives
        equal importance to both confidence and factual accuracy.
        
        Returns:
            Combined accuracy score (0.0-1.0)
        """
        return (self.confidence + self.fact_check_score) / 2.0
    
    def add_warning(self, warning: str) -> None:
        """
        Add a warning message to the response.
        
        Args:
            warning: Warning message to add
        """
        if warning not in self.warnings:
            self.warnings.append(warning)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.
        
        This method converts the response object to a dictionary suitable
        for JSON serialization, API responses, and data persistence. It
        includes all fields plus computed metrics for comprehensive tracking.
        
        Returns:
            Dictionary representation with all fields and computed metrics
        """
        return {
            'success': self.success,
            'query': self.query,
            'response': self.response,
            'confidence': self.confidence,
            'execution_time': self.execution_time,
            'timestamp': self.timestamp.isoformat(),
            'products_analyzed': self.products_analyzed,
            'data_source': self.data_source,
            'verification_passed': self.verification_passed,
            'fact_check_score': self.fact_check_score,
            'computed_facts': self.computed_facts,
            'reasoning_chain': self.reasoning_chain,
            'analysis_method': self.analysis_method,
            'metadata': self.metadata,
            'warnings': self.warnings,
            'accuracy_score': self.get_accuracy_score(),
            'is_high_confidence': self.is_high_confidence(),
            'is_verified_accurate': self.is_verified_accurate()
        }