"""
VIX Data Logger for Forward Testing
Records India VIX at each strategy run to build historical data for optimization
"""
import pandas as pd
import os
from datetime import datetime
import logging
import pytz


class VixLogger:
    """Logs VIX data at each strategy execution to build forward testing dataset"""
    
    def __init__(self, csv_path='logs/vix_history.csv'):
        """
        Initialize VIX logger
        
        Args:
            csv_path: Path to CSV file for VIX history
        """
        self.csv_path = csv_path
        self.ensure_file_exists()
    
    def ensure_file_exists(self):
        """Create VIX history file with headers if it doesn't exist"""
        if not os.path.exists(self.csv_path):
            # Create directory if needed
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            
            # Create file with headers
            df = pd.DataFrame(columns=[
                'timestamp',
                'date',
                'time',
                'weekday',
                'india_vix',
                'vix_condition',
                'vix_risk_level',
                'should_trade_per_vix',
                'vix_recommendation',
                'oi_analysis_done',
                'trade_signal',
                'trade_executed',
                'entry_price',
                'exit_price',
                'pnl',
                'pnl_percent',
                'outcome',
                'notes'
            ])
            df.to_csv(self.csv_path, index=False)
            logging.info(f"Created new VIX history file: {self.csv_path}")
    
    def log_vix_data(self, india_vix, vix_analysis, oi_done=False, trade_signal=None, 
                     trade_executed=False, entry_price=None, exit_price=None, 
                     pnl=None, pnl_percent=None, notes=''):
        """
        Log VIX data and strategy execution details
        
        Args:
            india_vix: Current India VIX value
            vix_analysis: Dict from get_vix_market_condition()
            oi_done: Whether OI analysis was performed
            trade_signal: 'LONG' or 'SHORT' or None
            trade_executed: Whether trade was actually taken
            entry_price: Trade entry price (if executed)
            exit_price: Trade exit price (if executed)
            pnl: Trade P&L (if executed)
            pnl_percent: Trade P&L percentage (if executed)
            notes: Any additional notes about the run
        """
        try:
            ist = pytz.timezone('Asia/Kolkata')
            now = datetime.now(ist)
            
            # Determine outcome
            outcome = ''
            if trade_executed and pnl is not None:
                if pnl > 0:
                    outcome = 'WIN'
                elif pnl < 0:
                    outcome = 'LOSS'
                else:
                    outcome = 'BREAKEVEN'
            elif trade_signal and not trade_executed:
                outcome = 'NO_TRADE'
            elif not oi_done:
                outcome = 'NO_OI_ANALYSIS'
            
            # Create record
            record = {
                'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
                'date': now.strftime('%Y-%m-%d'),
                'time': now.strftime('%H:%M:%S'),
                'weekday': now.strftime('%A'),
                'india_vix': round(india_vix, 2) if india_vix else None,
                'vix_condition': vix_analysis.get('condition', '') if vix_analysis else '',
                'vix_risk_level': vix_analysis.get('risk_level', '') if vix_analysis else '',
                'should_trade_per_vix': vix_analysis.get('should_trade', None) if vix_analysis else None,
                'vix_recommendation': vix_analysis.get('recommendation', '') if vix_analysis else '',
                'oi_analysis_done': oi_done,
                'trade_signal': trade_signal if trade_signal else '',
                'trade_executed': trade_executed,
                'entry_price': round(entry_price, 2) if entry_price else None,
                'exit_price': round(exit_price, 2) if exit_price else None,
                'pnl': round(pnl, 2) if pnl else None,
                'pnl_percent': round(pnl_percent, 2) if pnl_percent else None,
                'outcome': outcome,
                'notes': notes
            }
            
            # Append to CSV
            df = pd.DataFrame([record])
            
            # Use atomic write with temp file
            tmp_path = self.csv_path + '.tmp'
            
            # Read existing data
            if os.path.exists(self.csv_path):
                existing_df = pd.read_csv(self.csv_path)
                df = pd.concat([existing_df, df], ignore_index=True)
            
            # Write to temp file then rename
            df.to_csv(tmp_path, index=False)
            if os.path.exists(self.csv_path):
                os.remove(self.csv_path)
            os.replace(tmp_path, self.csv_path)
            
            logging.info(f"✓ VIX data logged: VIX={india_vix:.2f}, Condition={vix_analysis.get('condition', 'N/A') if vix_analysis else 'N/A'}, Outcome={outcome}")
            
        except Exception as e:
            logging.error(f"Error logging VIX data: {e}")
    
    def get_vix_statistics(self, min_records=10):
        """
        Get statistics from VIX history for optimization
        
        Returns dict with:
            - Win rate by VIX range
            - Avg P&L by VIX range
            - Best performing VIX ranges
            - Worst performing VIX ranges
        """
        try:
            if not os.path.exists(self.csv_path):
                return {"error": "No VIX history data available"}
            
            df = pd.read_csv(self.csv_path)
            
            if len(df) < min_records:
                return {
                    "error": f"Insufficient data. Need {min_records} records, have {len(df)}",
                    "current_records": len(df)
                }
            
            # Filter only executed trades
            trades_df = df[df['trade_executed'] == True].copy()
            
            if len(trades_df) < 5:
                return {
                    "error": f"Insufficient trade data. Need 5+ executed trades, have {len(trades_df)}",
                    "total_records": len(df),
                    "executed_trades": len(trades_df)
                }
            
            # Define VIX ranges
            vix_ranges = [
                (0, 10, "VIX <10 (Extreme Low)"),
                (10, 15, "VIX 10-15 (Low/Choppy)"),
                (15, 20, "VIX 15-20 (Normal)"),
                (20, 25, "VIX 20-25 (Elevated)"),
                (25, 30, "VIX 25-30 (High)"),
                (30, 35, "VIX 30-35 (Very High)"),
                (35, 100, "VIX >35 (Extreme High)")
            ]
            
            results = {}
            
            for low, high, label in vix_ranges:
                range_trades = trades_df[(trades_df['india_vix'] >= low) & (trades_df['india_vix'] < high)]
                
                if len(range_trades) > 0:
                    wins = len(range_trades[range_trades['pnl'] > 0])
                    losses = len(range_trades[range_trades['pnl'] < 0])
                    win_rate = (wins / len(range_trades)) * 100 if len(range_trades) > 0 else 0
                    avg_pnl = range_trades['pnl'].mean()
                    total_pnl = range_trades['pnl'].sum()
                    
                    results[label] = {
                        'trades': len(range_trades),
                        'wins': wins,
                        'losses': losses,
                        'win_rate': round(win_rate, 2),
                        'avg_pnl': round(avg_pnl, 2),
                        'total_pnl': round(total_pnl, 2),
                        'avg_vix': round(range_trades['india_vix'].mean(), 2)
                    }
            
            # Find best and worst ranges
            if results:
                sorted_by_winrate = sorted(results.items(), key=lambda x: x[1]['win_rate'], reverse=True)
                sorted_by_pnl = sorted(results.items(), key=lambda x: x[1]['total_pnl'], reverse=True)
                
                return {
                    "total_records": len(df),
                    "total_trades": len(trades_df),
                    "overall_win_rate": round((len(trades_df[trades_df['pnl'] > 0]) / len(trades_df)) * 100, 2),
                    "overall_avg_pnl": round(trades_df['pnl'].mean(), 2),
                    "overall_total_pnl": round(trades_df['pnl'].sum(), 2),
                    "by_vix_range": results,
                    "best_by_winrate": sorted_by_winrate[0] if sorted_by_winrate else None,
                    "best_by_pnl": sorted_by_pnl[0] if sorted_by_pnl else None,
                    "worst_by_winrate": sorted_by_winrate[-1] if sorted_by_winrate else None,
                    "worst_by_pnl": sorted_by_pnl[-1] if sorted_by_pnl else None
                }
            
            return {"error": "No data in any VIX range"}
            
        except Exception as e:
            logging.error(f"Error calculating VIX statistics: {e}")
            return {"error": str(e)}
    
    def print_vix_report(self):
        """Print a formatted report of VIX statistics"""
        stats = self.get_vix_statistics()
        
        if "error" in stats:
            print(f"\n⚠️ {stats['error']}")
            if "current_records" in stats:
                print(f"Current records: {stats['current_records']}")
            return
        
        print("\n" + "="*80)
        print("VIX FORWARD TESTING REPORT")
        print("="*80)
        print(f"Total Strategy Runs: {stats['total_records']}")
        print(f"Total Trades Executed: {stats['total_trades']}")
        print(f"Overall Win Rate: {stats['overall_win_rate']}%")
        print(f"Overall Avg P&L: ₹{stats['overall_avg_pnl']}")
        print(f"Overall Total P&L: ₹{stats['overall_total_pnl']}")
        print("\n" + "-"*80)
        print("PERFORMANCE BY VIX RANGE")
        print("-"*80)
        print(f"{'VIX Range':<25} {'Trades':<8} {'Win%':<8} {'Avg P&L':<12} {'Total P&L':<12} {'Avg VIX':<10}")
        print("-"*80)
        
        for vix_range, data in stats['by_vix_range'].items():
            print(f"{vix_range:<25} {data['trades']:<8} {data['win_rate']:<8.1f} ₹{data['avg_pnl']:<11.2f} ₹{data['total_pnl']:<11.2f} {data['avg_vix']:<10.2f}")
        
        print("\n" + "-"*80)
        print("KEY FINDINGS")
        print("-"*80)
        
        if stats['best_by_winrate']:
            best_wr = stats['best_by_winrate']
            print(f"✓ Best Win Rate: {best_wr[0]} - {best_wr[1]['win_rate']}% ({best_wr[1]['trades']} trades)")
        
        if stats['best_by_pnl']:
            best_pnl = stats['best_by_pnl']
            print(f"✓ Best Total P&L: {best_pnl[0]} - ₹{best_pnl[1]['total_pnl']} ({best_pnl[1]['trades']} trades)")
        
        if stats['worst_by_winrate']:
            worst_wr = stats['worst_by_winrate']
            print(f"✗ Worst Win Rate: {worst_wr[0]} - {worst_wr[1]['win_rate']}% ({worst_wr[1]['trades']} trades)")
        
        if stats['worst_by_pnl']:
            worst_pnl = stats['worst_by_pnl']
            print(f"✗ Worst Total P&L: {worst_pnl[0]} - ₹{worst_pnl[1]['total_pnl']} ({worst_pnl[1]['trades']} trades)")
        
        print("="*80 + "\n")


if __name__ == "__main__":
    # Test VIX logger
    from fetch_india_vix import fetch_india_vix, get_vix_market_condition
    
    logger = VixLogger()
    
    # Fetch current VIX
    vix = fetch_india_vix()
    if vix:
        analysis = get_vix_market_condition(vix)
        
        # Log sample data
        logger.log_vix_data(
            india_vix=vix,
            vix_analysis=analysis,
            oi_done=True,
            trade_signal='LONG',
            trade_executed=True,
            entry_price=100.50,
            exit_price=105.25,
            pnl=475.00,
            pnl_percent=4.73,
            notes='Test entry'
        )
        
        print(f"✓ Logged VIX {vix:.2f} to {logger.csv_path}")
        
        # Show statistics if enough data
        logger.print_vix_report()
    else:
        print("✗ Could not fetch VIX")
