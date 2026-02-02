"""
ROYALEY - Master Data Service
Unification layer that resolves teams, players, games, and odds across 27 sources.
"""

from app.services.master_data.master_data_service import MasterDataService
from app.services.master_data.feature_extractor import MasterFeatureExtractor, GameFeatureVector

__all__ = ["MasterDataService", "MasterFeatureExtractor", "GameFeatureVector"]