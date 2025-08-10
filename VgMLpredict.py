"""
Cardfight!! Vanguard D Series ML Predictor
==========================================
A machine learning system for predicting deck win rates and card prices
in the OverDress & Will+Dress (D Series) format.

Features:
- Deck win rate prediction based on D Series mechanics
- Card price prediction in Thai Baht
- Comprehensive data visualization
- D Series specific rules (exactly 1 Over Trigger per deck)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

# Machine Learning imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

class VanguardDSeriesMLPredictor:
    """
    Main class for Vanguard D Series predictions
    """

    def __init__(self):
        """Initialize the predictor with empty models and scalers"""
        self.deck_model = None
        self.price_model = None
        self.scaler_deck = StandardScaler()
        self.scaler_price = StandardScaler()
        self.label_encoders = {}

        # D Series constants
        self.NATIONS = [
            'Dragon Empire', 'Dark States', 'Brandt Gate',
            'Keter Sanctuary', 'Stoicheia', 'Lyrical Monasterio'
        ]

        self.RIDE_LINES = [
            'Blaster Blade', 'Dragonic Overlord', 'Machining', 'Sealed Blaze Maiden',
            'Aurora Battle Princess', 'Magnolia', 'Sylvan Horned Beast', 'Elementaria',
            'Jewel Knights', 'Earnescorrect', 'Monster Strike', 'Peach Sisters'
        ]

        self.CARD_TYPES = [
            'Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Over Trigger',
            'Normal Order', 'Blitz Order', 'Set Order'
        ]

        self.RARITIES = ['C', 'R', 'RR', 'RRR', 'VR', 'SVR', 'UR', 'LR', 'OR']

    # ============================================================================
    # DATA GENERATION METHODS
    # ============================================================================

    def generate_d_series_data(self):
        """
        Generate realistic D Series sample data for training

        Creates two datasets:
        1. Deck data - for win rate prediction
        2. Card data - for price prediction
        """
        print("Generating D Series sample data...")
        np.random.seed(42)

        # Generate deck data
        self.deck_df = self._generate_deck_data()

        # Generate card data
        self.card_df = self._generate_card_data()

        print(f"✅ Sample data generated successfully!")
        print(f"📊 Deck data: {self.deck_df.shape[0]} samples")
        print(f"💳 Card data: {self.card_df.shape[0]} samples")

    def _generate_deck_data(self):
        """Generate realistic deck composition data"""
        deck_data = []

        for i in range(800):  # Generate 800 deck samples
            # Basic deck info
            nation = np.random.choice(self.NATIONS)
            ride_line = np.random.choice(self.RIDE_LINES)

            # D Series deck composition (50 cards total)
            deck_composition = self._generate_deck_composition()

            # D Series specific mechanics
            mechanics = self._generate_deck_mechanics()

            # Calculate win rate based on D Series factors
            win_rate = self._calculate_deck_win_rate(deck_composition, mechanics)

            # Combine all data
            deck_entry = {
                'nation': nation,
                'ride_line': ride_line,
                'avg_card_cost': np.random.normal(18, 10),
                'tournament_top_cuts': np.random.randint(0, 30),
                'win_rate': win_rate,
                **deck_composition,
                **mechanics
            }

            deck_data.append(deck_entry)

        return pd.DataFrame(deck_data)

    def _generate_deck_composition(self):
        """Generate realistic card count distribution for D Series"""
        return {
            'grade0_count': np.random.randint(16, 18),     # First Vanguard + triggers
            'grade1_count': np.random.randint(12, 16),     # Grade 1 units
            'grade2_count': np.random.randint(10, 14),     # Grade 2 units
            'grade3_count': np.random.randint(6, 10),      # Grade 3 units
            'over_trigger_count': 1,                        # Exactly 1 Over Trigger (D Series rule)
            'normal_order_count': np.random.randint(0, 6), # Normal Orders
            'blitz_order_count': np.random.randint(0, 4),  # Blitz Orders
            'set_order_count': np.random.randint(0, 3)     # Set Orders
        }

    def _generate_deck_mechanics(self):
        """Generate D Series specific mechanics scores"""
        return {
            'persona_ride_chance': np.random.uniform(0.6, 0.95),           # Persona Ride consistency
            'ride_line_synergy': max(0, min(100, np.random.normal(75, 12))), # Ride line synergy (0-100)
            'over_trigger_synergy': max(0, min(100, np.random.normal(70, 15))), # Over Trigger optimization
            'meta_relevance': max(0, min(10, np.random.normal(5, 2.5)))    # Meta relevance (0-10)
        }

    def _calculate_deck_win_rate(self, composition, mechanics):
        """Calculate realistic win rate based on D Series mechanics"""
        base_win_rate = (
            mechanics['ride_line_synergy'] * 0.3 +      # 30% weight on synergy
            mechanics['persona_ride_chance'] * 30 +      # 30% weight on consistency
            mechanics['over_trigger_synergy'] * 0.2 +    # 20% weight on Over Trigger
            mechanics['meta_relevance'] * 6 +            # Meta relevance factor
            np.random.normal(0, 8)                       # Random variance
        ) / 100

        return max(0.25, min(0.90, base_win_rate))  # Clamp between 25% and 90%

    def _generate_card_data(self):
        """Generate realistic card price data"""
        card_data = []

        for i in range(400):  # Generate 400 card samples
            # Basic card info
            nation = np.random.choice(self.NATIONS)
            card_type = np.random.choice(self.CARD_TYPES)
            ride_line = np.random.choice(self.RIDE_LINES)

            # Card rarity (weighted distribution)
            rarity = np.random.choice(
                self.RARITIES,
                p=[0.35, 0.25, 0.18, 0.12, 0.05, 0.02, 0.015, 0.01, 0.005]
            )

            # Market factors
            market_factors = self._generate_market_factors()

            # D Series specific factors
            is_ride_line_key = np.random.choice([True, False], p=[0.3, 0.7])
            is_over_trigger = card_type == 'Over Trigger'
            competitive_score = max(0, min(10, np.random.normal(5.5, 2)))

            # Calculate realistic Thai Baht price
            price = self._calculate_card_price(rarity, market_factors, is_ride_line_key,
                                             is_over_trigger, competitive_score)

            card_entry = {
                'nation': nation,
                'card_type': card_type,
                'ride_line': ride_line,
                'rarity': rarity,
                'competitive_score': competitive_score,
                'is_ride_line_key': is_ride_line_key,
                'is_over_trigger': is_over_trigger,
                'price': price,
                **market_factors
            }

            card_data.append(card_entry)

        return pd.DataFrame(card_data)

    def _generate_market_factors(self):
        """Generate market-related factors affecting card prices"""
        return {
            'tournament_usage': min(100, np.random.exponential(3)),    # Tournament usage %
            'days_since_release': np.random.randint(1, 800),          # Days since release
            'supply': np.random.exponential(40)                       # Market supply
        }

    def _calculate_card_price(self, rarity, market_factors, is_ride_line_key,
                             is_over_trigger, competitive_score):
        """
        Calculate realistic card price in Thai Baht based on:
        - C (Common): ~10฿
        - R (Rare): ~15฿
        - RR (Double Rare): ~20฿
        - RRR (Triple Rare): ~20-35฿
        - SP/SVR/GR/SGR: ~150-900฿
        """
        # Base prices for different rarities
        if rarity == 'C':
            base_price = 10
        elif rarity == 'R':
            base_price = 15
        elif rarity == 'RR':
            base_price = 20
        elif rarity == 'RRR':
            # RRR ranges from 20-35฿
            base_price = np.random.uniform(20, 35) * (1 + market_factors['tournament_usage']/20)
        else:
            # Premium cards (VR, SVR, UR, LR, OR) range 150-900฿
            base_price = np.random.uniform(150, 900) * (1 + market_factors['tournament_usage']/12)

        # Apply market modifiers
        price_modifier = (
            (500 / (market_factors['days_since_release'] + 100)) *  # Newer = more expensive
            (50 / max(market_factors['supply'], 1))                 # Low supply = higher price
        )

        final_price = base_price * price_modifier

        # D Series specific bonuses
        if is_ride_line_key:
            final_price *= 1.8  # +80% for key cards
        if is_over_trigger and competitive_score > 7:
            final_price *= 2.2  # +120% for competitive Over Triggers

        # Add realistic variance
        if rarity in ['C', 'R', 'RR']:
            variance = final_price * 0.15  # Low variance for commons
        else:
            variance = final_price * 0.3   # High variance for premium cards

        final_price += np.random.normal(0, variance)

        return max(5, final_price)  # Minimum ฿5

    # ============================================================================
    # DATA PREPROCESSING METHODS
    # ============================================================================

    def preprocess_d_series_data(self):
        """
        Preprocess the generated data for machine learning
        - Encode categorical variables
        - Select relevant features
        - Prepare training datasets
        """
        print("Preprocessing D Series data...")

        # Process deck data for win rate prediction
        self._preprocess_deck_data()

        # Process card data for price prediction
        self._preprocess_card_data()

        print("✅ Data preprocessing completed!")

    def _preprocess_deck_data(self):
        """Prepare deck data for machine learning"""
        deck_features = self.deck_df.copy()

        # Encode categorical variables
        for col in ['nation', 'ride_line']:
            le = LabelEncoder()
            deck_features[col + '_encoded'] = le.fit_transform(deck_features[col])
            self.label_encoders[f'deck_{col}'] = le

        # Define features for deck win rate prediction
        self.deck_features = [
            'nation_encoded', 'ride_line_encoded', 'grade0_count', 'grade1_count',
            'grade2_count', 'grade3_count', 'over_trigger_count', 'normal_order_count',
            'blitz_order_count', 'set_order_count', 'persona_ride_chance',
            'ride_line_synergy', 'over_trigger_synergy', 'meta_relevance',
            'avg_card_cost', 'tournament_top_cuts'
        ]

        self.X_deck = deck_features[self.deck_features]
        self.y_deck = deck_features['win_rate']

    def _preprocess_card_data(self):
        """Prepare card data for machine learning"""
        card_features = self.card_df.copy()

        # Encode categorical variables
        for col in ['nation', 'card_type', 'ride_line', 'rarity']:
            le = LabelEncoder()
            card_features[col + '_encoded'] = le.fit_transform(card_features[col])
            self.label_encoders[f'card_{col}'] = le

        # Convert boolean features to numeric
        card_features['is_ride_line_key_num'] = card_features['is_ride_line_key'].astype(int)
        card_features['is_over_trigger_num'] = card_features['is_over_trigger'].astype(int)

        # Define features for card price prediction
        self.card_features = [
            'nation_encoded', 'card_type_encoded', 'ride_line_encoded', 'rarity_encoded',
            'tournament_usage', 'days_since_release', 'supply', 'competitive_score',
            'is_ride_line_key_num', 'is_over_trigger_num'
        ]

        self.X_card = card_features[self.card_features]
        self.y_card = card_features['price']

    # ============================================================================
    # MODEL TRAINING METHODS
    # ============================================================================

    def train_d_series_models(self):
        """
        Train both the deck win rate and card price prediction models
        Returns performance metrics for both models
        """
        print("Training D Series models...")

        # Train deck win rate model
        deck_metrics = self._train_deck_model()

        # Train card price model
        price_metrics = self._train_price_model()

        return deck_metrics + price_metrics

    def _train_deck_model(self):
        """Train the deck win rate prediction model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_deck, self.y_deck, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler_deck.fit_transform(X_train)
        X_test_scaled = self.scaler_deck.transform(X_test)

        # Train Gradient Boosting model
        self.deck_model = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=7,
            random_state=42
        )
        self.deck_model.fit(X_train_scaled, y_train)

        # Evaluate model
        y_pred = self.deck_model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"🎯 Deck Win Rate Model Performance:")
        print(f"   Mean Absolute Error: {mae:.4f}")
        print(f"   R² Score: {r2:.4f}")

        return mae, r2

    def _train_price_model(self):
        """Train the card price prediction model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_card, self.y_card, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler_price.fit_transform(X_train)
        X_test_scaled = self.scaler_price.transform(X_test)

        # Train Random Forest model
        self.price_model = RandomForestRegressor(
            n_estimators=120,
            max_depth=12,
            random_state=42
        )
        self.price_model.fit(X_train_scaled, y_train)

        # Evaluate model
        y_pred = self.price_model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"💰 Card Price Model Performance:")
        print(f"   Mean Absolute Error: ฿{mae:.2f}")
        print(f"   R² Score: {r2:.4f}")

        return mae, r2

    # ============================================================================
    # PREDICTION METHODS
    # ============================================================================

    def predict_deck_winrate(self, nation, ride_line, grade0_count, grade1_count,
                           grade2_count, grade3_count, normal_order_count,
                           blitz_order_count, set_order_count, persona_ride_chance,
                           ride_line_synergy, over_trigger_synergy, meta_relevance,
                           avg_card_cost, tournament_top_cuts):
        """
        Predict win rate for a D Series deck
        Over Trigger count is automatically set to 1 (D Series rule)
        """
        if self.deck_model is None:
            raise ValueError("❌ Deck model not trained yet! Call train_d_series_models() first.")

        # Encode categorical variables
        nation_encoded = self.label_encoders['deck_nation'].transform([nation])[0]
        ride_line_encoded = self.label_encoders['deck_ride_line'].transform([ride_line])[0]

        # Create feature array (over_trigger_count is always 1)
        features = np.array([[
            nation_encoded, ride_line_encoded, grade0_count, grade1_count,
            grade2_count, grade3_count, 1, normal_order_count,  # 1 = over_trigger_count
            blitz_order_count, set_order_count, persona_ride_chance,
            ride_line_synergy, over_trigger_synergy, meta_relevance,
            avg_card_cost, tournament_top_cuts
        ]])

        # Scale and predict
        features_scaled = self.scaler_deck.transform(features)
        prediction = self.deck_model.predict(features_scaled)[0]

        return max(0, min(1, prediction))  # Ensure 0-1 range

    def predict_card_price(self, nation, card_type, ride_line, rarity, tournament_usage,
                         days_since_release, supply, competitive_score, is_ride_line_key,
                         is_over_trigger):
        """Predict price for a D Series card in Thai Baht"""
        if self.price_model is None:
            raise ValueError("❌ Price model not trained yet! Call train_d_series_models() first.")

        # Encode categorical variables
        nation_encoded = self.label_encoders['card_nation'].transform([nation])[0]
        card_type_encoded = self.label_encoders['card_card_type'].transform([card_type])[0]
        ride_line_encoded = self.label_encoders['card_ride_line'].transform([ride_line])[0]
        rarity_encoded = self.label_encoders['card_rarity'].transform([rarity])[0]

        # Create feature array
        features = np.array([[
            nation_encoded, card_type_encoded, ride_line_encoded, rarity_encoded,
            tournament_usage, days_since_release, supply, competitive_score,
            int(is_ride_line_key), int(is_over_trigger)
        ]])

        # Scale and predict
        features_scaled = self.scaler_price.transform(features)
        prediction = self.price_model.predict(features_scaled)[0]

        return max(5, prediction)  # Minimum ฿5

    # ============================================================================
    # ANALYSIS METHODS
    # ============================================================================

    def analyze_d_series_meta(self):
        """Comprehensive analysis of D Series meta performance"""
        print("\n" + "="*60)
        print("📊 D SERIES META ANALYSIS")
        print("="*60)

        # Nation performance analysis
        nation_stats = self.deck_df.groupby('nation').agg({
            'win_rate': ['mean', 'std', 'count'],
            'meta_relevance': 'mean',
            'ride_line_synergy': 'mean',
            'persona_ride_chance': 'mean'
        }).round(4)

        print("\n🏛️ Nation Performance:")
        print(nation_stats)

        # Top ride lines analysis
        ride_line_stats = self.deck_df.groupby('ride_line').agg({
            'win_rate': 'mean',
            'tournament_top_cuts': 'mean'
        }).round(4).sort_values('win_rate', ascending=False)

        print("\n🚀 Top Ride Lines by Win Rate:")
        print(ride_line_stats.head(10))

        # Over Trigger synergy analysis
        print("\n⚡ Over Trigger Synergy Analysis:")
        print("All decks have exactly 1 Over Trigger (D Series rule)")
        print("Higher synergy scores indicate better Over Trigger selection:")

        return nation_stats, ride_line_stats

    def analyze_card_prices(self):
        """Analysis of D Series card price trends"""
        print("\n" + "="*60)
        print("💰 D SERIES PRICE ANALYSIS")
        print("="*60)

        # Price by rarity analysis
        rarity_prices = self.card_df.groupby('rarity').agg({
            'price': ['mean', 'median', 'std'],
            'tournament_usage': 'mean'
        }).round(2)

        print("\n💎 Average Prices by Rarity (Thai Baht):")
        print(rarity_prices)

        # Most expensive Over Triggers
        print("\n⚡ Most Expensive Over Triggers:")
        over_triggers = self.card_df[self.card_df['is_over_trigger'] == True].nlargest(5, 'price')
        print(over_triggers[['nation', 'rarity', 'competitive_score', 'price']].round(2))

        # Most expensive ride line key cards
        print("\n🔑 Most Expensive Ride Line Key Cards:")
        ride_line_keys = self.card_df[self.card_df['is_ride_line_key'] == True].nlargest(5, 'price')
        print(ride_line_keys[['nation', 'ride_line', 'rarity', 'price']].round(2))

        return rarity_prices

    # ============================================================================
    # VISUALIZATION METHODS
    # ============================================================================

    def create_all_visualizations(self):
        """Generate all visualization charts"""
        print("\n" + "="*60)
        print("📈 GENERATING VISUALIZATIONS...")
        print("="*60)

        self.plot_feature_importance()
        self.plot_nation_performance()
        self.plot_correlation_matrix()
        self.plot_price_trends()

    def plot_feature_importance(self):
        """Plot feature importance for both models"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Deck model feature importance
        if self.deck_model is not None:
            importance_deck = self.deck_model.feature_importances_
            feature_names_deck = [name.replace('_encoded', '').replace('_', ' ').title()
                                 for name in self.deck_features]

            # Sort by importance
            sorted_indices = np.argsort(importance_deck)[::-1]
            sorted_importance = importance_deck[sorted_indices]
            sorted_names = [feature_names_deck[i] for i in sorted_indices]

            bars1 = ax1.barh(range(len(sorted_names)), sorted_importance,
                           color='skyblue', alpha=0.8)
            ax1.set_yticks(range(len(sorted_names)))
            ax1.set_yticklabels(sorted_names)
            ax1.set_title('🎯 Deck Win Rate - Feature Importance', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Importance Score', fontsize=12)
            ax1.grid(axis='x', alpha=0.3)

            # Add value labels
            for i, bar in enumerate(bars1):
                width = bar.get_width()
                ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                        f'{width:.3f}', ha='left', va='center', fontsize=10)

        # Price model feature importance
        if self.price_model is not None:
            importance_price = self.price_model.feature_importances_
            feature_names_price = [name.replace('_encoded', '').replace('_', ' ').title().replace('Num', '')
                                  for name in self.card_features]

            sorted_indices = np.argsort(importance_price)[::-1]
            sorted_importance = importance_price[sorted_indices]
            sorted_names = [feature_names_price[i] for i in sorted_indices]

            bars2 = ax2.barh(range(len(sorted_names)), sorted_importance,
                           color='lightcoral', alpha=0.8)
            ax2.set_yticks(range(len(sorted_names)))
            ax2.set_yticklabels(sorted_names)
            ax2.set_title('💰 Card Price (฿) - Feature Importance', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Importance Score', fontsize=12)
            ax2.grid(axis='x', alpha=0.3)

            for i, bar in enumerate(bars2):
                width = bar.get_width()
                ax2.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                        f'{width:.3f}', ha='left', va='center', fontsize=10)

        plt.tight_layout()
        plt.show()

    def plot_nation_performance(self):
        """Plot comprehensive nation performance dashboard"""
        nation_stats = self.deck_df.groupby('nation').agg({
            'win_rate': 'mean',
            'meta_relevance': 'mean',
            'persona_ride_chance': 'mean'
        }).round(3)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        nations = nation_stats.index

        # 1. Win rate by nation
        win_rates = nation_stats['win_rate']
        bars1 = ax1.bar(nations, win_rates, color='lightgreen', alpha=0.8)
        ax1.set_title('🏆 Average Win Rate by Nation', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Win Rate')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)

        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom')

        # 2. Meta relevance by nation
        meta_relevance = nation_stats['meta_relevance']
        bars2 = ax2.bar(nations, meta_relevance, color='orange', alpha=0.8)
        ax2.set_title('📈 Meta Relevance by Nation', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Meta Relevance Score')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)

        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.2f}', ha='center', va='bottom')

        # 3. Persona ride consistency
        persona_ride = nation_stats['persona_ride_chance']
        bars3 = ax3.bar(nations, persona_ride, color='purple', alpha=0.8)
        ax3.set_title('🎭 Persona Ride Consistency by Nation', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Persona Ride Chance')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(axis='y', alpha=0.3)

        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')

        # 4. Price distribution by rarity
        rarity_prices = self.card_df.groupby('rarity')['price'].mean().sort_values(ascending=False)
        bars4 = ax4.bar(rarity_prices.index, rarity_prices.values, color='gold', alpha=0.8)
        ax4.set_title('💰 Average Card Price by Rarity (฿)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Price (Thai Baht)')
        ax4.grid(axis='y', alpha=0.3)

        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 10,
                    f'฿{height:.0f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    def plot_correlation_matrix(self):
        """Plot correlation matrix for deck features"""
        numerical_cols = [
            'grade0_count', 'grade1_count', 'grade2_count', 'grade3_count',
            'persona_ride_chance', 'ride_line_synergy', 'over_trigger_synergy',
            'meta_relevance', 'tournament_top_cuts', 'win_rate'
        ]

        correlation_matrix = self.deck_df[numerical_cols].corr()

        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

        # Create heatmap
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r',
                   center=0, square=True, fmt='.3f', cbar_kws={"shrink": .8})

        plt.title('🔗 D Series Deck Features Correlation Matrix',
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()

    def plot_price_trends(self):
        """Plot comprehensive price analysis charts"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Price distribution histogram
        ax1.hist(self.card_df['price'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('💰 Card Price Distribution (฿)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Price (Thai Baht)')
        ax1.set_ylabel('Frequency')
        ax1.grid(axis='y', alpha=0.3)

        # 2. Price vs Tournament Usage
        ax2.scatter(self.card_df['tournament_usage'], self.card_df['price'],
                   alpha=0.6, color='green')
        ax2.set_title('🏆 Price vs Tournament Usage', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Tournament Usage (%)')
        ax2.set_ylabel('Price (฿)')
        ax2.grid(alpha=0.3)

        # 3. Price vs Competitive Score
        ax3.scatter(self.card_df['competitive_score'], self.card_df['price'],
                   alpha=0.6, color='red')
        ax3.set_title('⭐ Price vs Competitive Score', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Competitive Score')
        ax3.set_ylabel('Price (฿)')
        ax3.grid(alpha=0.3)

        # 4. Over Trigger vs Regular cards price comparison
        over_trigger_prices = self.card_df[self.card_df['is_over_trigger'] == True]['price']
        regular_prices = self.card_df[self.card_df['is_over_trigger'] == False]['price']

        ax4.boxplot([regular_prices, over_trigger_prices],
                   labels=['Regular Cards', 'Over Triggers'])
        ax4.set_title('⚡ Price Comparison: Regular vs Over Trigger Cards',
                     fontsize=14, fontweight='bold')
        ax4.set_ylabel('Price (฿)')
        ax4.set_ylim(0, 1000)  # Realistic price range
        ax4.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.show()


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """
    Main function to demonstrate the D Series ML Predictor
    """
    print("🚀 Starting Cardfight!! Vanguard D Series ML Predictor")
    print("="*60)

    # Initialize predictor
    predictor = VanguardDSeriesMLPredictor()

    # Step 1: Generate sample data
    predictor.generate_d_series_data()

    # Step 2: Preprocess data
    predictor.preprocess_d_series_data()

    # Step 3: Train models
    predictor.train_d_series_models()

    # Step 4: Example predictions
    print("\n" + "="*60)
    print("🔮 D SERIES EXAMPLE PREDICTIONS")
    print("="*60)

    # Example 1: Predict deck win rate
    print("\n📊 Example Deck Prediction:")
    deck_winrate = predictor.predict_deck_winrate(
        nation='Dragon Empire',
        ride_line='Dragonic Overlord',
        grade0_count=17,
        grade1_count=14,
        grade2_count=11,
        grade3_count=8,
        normal_order_count=2,
        blitz_order_count=1,
        set_order_count=0,
        persona_ride_chance=0.85,
        ride_line_synergy=88,
        over_trigger_synergy=75,
        meta_relevance=8.2,
        avg_card_cost=25.0,
        tournament_top_cuts=12
    )
    print(f"🎯 Predicted win rate for Dragonic Overlord deck: {deck_winrate:.3f} ({deck_winrate*100:.1f}%)")

    # Example 2: Predict premium card price
    print("\n💳 Example Card Price Predictions:")

    # Premium VR card
    card_price = predictor.predict_card_price(
        nation='Keter Sanctuary',
        card_type='Grade 3',
        ride_line='Blaster Blade',
        rarity='VR',
        tournament_usage=65,
        days_since_release=90,
        supply=15,
        competitive_score=9.2,
        is_ride_line_key=True,
        is_over_trigger=False
    )
    print(f"💎 Blaster Blade VR (Key Card): ฿{card_price:.2f}")

    # Over Trigger card
    over_trigger_price = predictor.predict_card_price(
        nation='Stoicheia',
        card_type='Over Trigger',
        ride_line='Magnolia',
        rarity='OR',
        tournament_usage=85,
        days_since_release=45,
        supply=8,
        competitive_score=9.5,
        is_ride_line_key=False,
        is_over_trigger=True
    )
    print(f"⚡ Stoicheia Over Trigger: ฿{over_trigger_price:.2f}")

    # Common card
    common_price = predictor.predict_card_price(
        nation='Dark States',
        card_type='Grade 1',
        ride_line='Machining',
        rarity='C',
        tournament_usage=15,
        days_since_release=200,
        supply=100,
        competitive_score=4.0,
        is_ride_line_key=False,
        is_over_trigger=False
    )
    print(f"🔧 Dark States Common Grade 1: ฿{common_price:.2f}")

    # Step 5: Meta analysis
    predictor.analyze_d_series_meta()
    predictor.analyze_card_prices()

    # Step 6: Generate visualizations
    predictor.create_all_visualizations()

    print("\n" + "="*60)
    print("✅ D Series ML Predictor completed successfully!")
    print("="*60)

    return predictor


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def quick_deck_prediction(predictor, deck_name, nation, ride_line, synergy_score=80):
    """
    Quick helper function for common deck predictions
    """
    winrate = predictor.predict_deck_winrate(
        nation=nation,
        ride_line=ride_line,
        grade0_count=17,
        grade1_count=14,
        grade2_count=11,
        grade3_count=8,
        normal_order_count=2,
        blitz_order_count=1,
        set_order_count=0,
        persona_ride_chance=0.80,
        ride_line_synergy=synergy_score,
        over_trigger_synergy=70,
        meta_relevance=6.0,
        avg_card_cost=20.0,
        tournament_top_cuts=8
    )

    print(f"🎯 {deck_name}: {winrate:.3f} ({winrate*100:.1f}% win rate)")
    return winrate


# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run the main program
    predictor = main()

    # Optional: Quick predictions for popular decks
    print("\n" + "="*60)
    print("🔥 QUICK PREDICTIONS FOR POPULAR DECKS")
    print("="*60)

    quick_deck_prediction(predictor, "Dragonic Overlord", "Dragon Empire", "Dragonic Overlord", 85)
    quick_deck_prediction(predictor, "Blaster Blade", "Keter Sanctuary", "Blaster Blade", 82)
    quick_deck_prediction(predictor, "Magnolia", "Stoicheia", "Magnolia", 78)
