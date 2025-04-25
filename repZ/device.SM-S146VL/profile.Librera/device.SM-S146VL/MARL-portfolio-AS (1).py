import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from transformers import AutoModel, AutoTokenizer


class MultiAssetAdaptiveRL:
    def __init__(self, config: Dict[str, Any]):
        """
        Multi-Asset Reinforcement Learning Framework with feature engineering and risk management.
        """
        self.config = {
            'assets': ['SPY', 'QQQ', 'AGG', 'GLD'],
            'feature_dimensions': 20,
            'learning_rate': 1e-3,
            'risk_tolerance': 0.05,
            **config,
        }
        self.feature_extractor = self._build_feature_extractor()
        self.adaptive_network = self._build_adaptive_network()
        self.optimizer = optim.Adam(self.adaptive_network.parameters(), lr=self.config['learning_rate'])
        self.risk_manager = RiskManagementModule(self.config)

    def _build_feature_extractor(self) -> nn.Module:
        """
        Builds a neural network for feature extraction.
        """
        return nn.Sequential(
            nn.Linear(50, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, self.config['feature_dimensions']),
            nn.LayerNorm(self.config['feature_dimensions']),
        )

    def _build_adaptive_network(self) -> nn.Module:
        """
        Builds the adaptive network for decision-making.
        """
        return nn.Sequential(
            nn.Linear(self.config['feature_dimensions'], 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, len(self.config['assets'])),
            nn.Softmax(dim=-1),
        )

    def preprocess_market_data(self, market_data: pd.DataFrame) -> torch.Tensor:
        """
        Processes market data into feature tensors for model input.
        """
        # Ensure correlation matrices are vectorized properly
        correlation_features = np.concatenate(
            [corr_matrix.flatten() for corr_matrix in market_data['correlation_matrix']]
        )
        features = np.hstack([
            market_data['returns'],
            market_data['volatility'],
            market_data['momentum'],
            market_data['volume_change'],
            correlation_features,
        ])
        features_tensor = torch.FloatTensor(features).unsqueeze(0)  # Add batch dimension
        return self.feature_extractor(features_tensor)

    def compute_portfolio_allocation(self, market_data: pd.DataFrame) -> np.ndarray:
        """
        Computes the portfolio allocation weights.
        """
        processed_features = self.preprocess_market_data(market_data)
        with torch.no_grad():  # Avoid gradient computations
            allocation_weights = self.adaptive_network(processed_features).squeeze().numpy()
        return self.risk_manager.constrain_allocation(allocation_weights)


class RiskManagementModule:
    def __init__(self, config: Dict[str, Any]):
        """
        Risk management module for enforcing allocation constraints.
        """
        self.config = config

    def constrain_allocation(self, weights: np.ndarray) -> np.ndarray:
        """
        Enforces allocation constraints on portfolio weights.
        """
        max_allocation = 0.4  # Max 40% per asset
        min_allocation = 0.05  # Min 5% per asset
        constrained_weights = np.clip(weights, min_allocation, max_allocation)
        return constrained_weights / constrained_weights.sum()


def integrate_with_huggingface(model_name: str = "9x25dillon/9xdSq-LIMPS-FemT0-R1C"):
    """
    Integrates the adaptive RL model with a Hugging Face Transformer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    transformer_model = AutoModel.from_pretrained(model_name)
    config = {
        'assets': ['SPY', 'QQQ', 'AGG', 'GLD'],
        'feature_dimensions': 20,
        'learning_rate': 5e-4,
        'risk_tolerance': 0.05,
    }
    strategy = MultiAssetAdaptiveRL(config)
    return {'strategy': strategy, 'transformer_model': transformer_model, 'tokenizer': tokenizer}


def main():
    """
    Entry point for executing the adaptive reinforcement learning pipeline.
    """
    integrated_model = integrate_with_huggingface()

    # Generate synthetic market data for testing
    np.random.seed(42)  # For reproducibility
    synthetic_market_data = pd.DataFrame({
        'returns': np.random.normal(0.001, 0.02, 100),
        'volatility': np.random.uniform(0.01, 0.1, 100),
        'momentum': np.random.normal(0, 0.015, 100),
        'volume_change': np.random.normal(0, 0.05, 100),
        'correlation_matrix': [np.random.rand(4, 4) for _ in range(100)],
    })

    # Compute portfolio allocation for first data sample
    sample_data = synthetic_market_data.iloc[0]
    portfolio_weights = integrated_model['strategy'].compute_portfolio_allocation(sample_data)
    
    print("Computed Portfolio Allocation:")
    for asset, weight in zip(integrated_model['strategy'].config['assets'], portfolio_weights):
        print(f"{asset}: {weight * 100:.2f}%")


if __name__ == "__main__":
    main()