"""
Analyze VIX Forward Testing Data
Run this after collecting VIX data to determine optimal VIX range for your strategy
"""
import pandas as pd
import os

def analyze_vix_performance():
    """Analyze trade performance across different VIX ranges"""
    
    csv_path = 'logs/trade_history.csv'
    
    if not os.path.exists(csv_path):
        return {"error": "Trade history file not found"}
    
    df = pd.read_csv(csv_path)
    
    # Check if VIX column exists
    if 'India VIX' not in df.columns:
        return {
            "error": "VIX data not found in trade history",
            "total_trades": len(df),
            "message": "Start trading with new code to collect VIX data"
        }
    
    # Filter trades with VIX data
    trades_with_vix = df[df['India VIX'].notna()].copy()
    
    if len(trades_with_vix) < 5:
        return {
            "error": f"Insufficient trades with VIX data. Need 5+, have {len(trades_with_vix)}",
            "total_trades": len(df),
            "trades_with_vix": len(trades_with_vix)
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
        range_trades = trades_with_vix[
            (trades_with_vix['India VIX'] >= low) & 
            (trades_with_vix['India VIX'] < high)
        ].copy()
        
        if len(range_trades) > 0:
            wins = len(range_trades[range_trades['P&L'] > 0])
            losses = len(range_trades[range_trades['P&L'] < 0])
            win_rate = (wins / len(range_trades)) * 100 if len(range_trades) > 0 else 0
            avg_pnl = range_trades['P&L'].mean()
            total_pnl = range_trades['P&L'].sum()
            
            results[label] = {
                'trades': len(range_trades),
                'wins': wins,
                'losses': losses,
                'win_rate': round(win_rate, 2),
                'avg_pnl': round(avg_pnl, 2),
                'total_pnl': round(total_pnl, 2),
                'avg_vix': round(range_trades['India VIX'].mean(), 2)
            }
    
    # Overall stats
    overall_wins = len(trades_with_vix[trades_with_vix['P&L'] > 0])
    overall_win_rate = (overall_wins / len(trades_with_vix)) * 100
    overall_avg_pnl = trades_with_vix['P&L'].mean()
    overall_total_pnl = trades_with_vix['P&L'].sum()
    
    # Find best and worst ranges
    if results:
        sorted_by_winrate = sorted(results.items(), key=lambda x: x[1]['win_rate'], reverse=True)
        sorted_by_pnl = sorted(results.items(), key=lambda x: x[1]['total_pnl'], reverse=True)
        
        return {
            "total_trades": len(trades_with_vix),
            "overall_win_rate": round(overall_win_rate, 2),
            "overall_avg_pnl": round(overall_avg_pnl, 2),
            "overall_total_pnl": round(overall_total_pnl, 2),
            "by_vix_range": results,
            "best_by_winrate": sorted_by_winrate[0] if sorted_by_winrate else None,
            "best_by_pnl": sorted_by_pnl[0] if sorted_by_pnl else None,
            "worst_by_winrate": sorted_by_winrate[-1] if sorted_by_winrate else None,
            "worst_by_pnl": sorted_by_pnl[-1] if sorted_by_pnl else None
        }
    
    return {"error": "No data in any VIX range"}

def main():
    print("\n" + "="*80)
    print("VIX FORWARD TESTING ANALYSIS")
    print("="*80)
    print("\nThis tool analyzes your collected VIX data to find optimal trading conditions")
    print("based on YOUR strategy's actual performance across different VIX ranges.\n")
    
    # Get statistics
    stats = analyze_vix_performance()
    
    if "error" not in stats:
        print(f"\nTotal Trades Analyzed: {stats['total_trades']}")
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
        
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)
        
        # Find ranges with positive total P&L and win rate >= 50%
        profitable_ranges = []
        for vix_range, data in stats['by_vix_range'].items():
            if data['total_pnl'] > 0 and data['win_rate'] >= 50 and data['trades'] >= 3:
                profitable_ranges.append((vix_range, data))
        
        if profitable_ranges:
            print("\n✓ PROFITABLE VIX RANGES (Win Rate ≥50%, Total P&L >0, Min 3 trades):")
            for vix_range, data in profitable_ranges:
                print(f"  • {vix_range}: {data['win_rate']}% win rate, ₹{data['total_pnl']} total P&L ({data['trades']} trades)")
            
            # Extract VIX min/max from profitable ranges
            vix_values = []
            for vix_range, data in profitable_ranges:
                vix_values.append(data['avg_vix'])
            
            if vix_values:
                suggested_min = max(10, min(vix_values) - 2)  # Buffer, but not below 10
                suggested_max = min(35, max(vix_values) + 2)  # Buffer, but not above 35
                
                print("\n" + "-"*80)
                print("SUGGESTED CONFIG.YAML SETTINGS:")
                print("-"*80)
                print(f"vix_min_threshold: {suggested_min:.1f}")
                print(f"vix_max_threshold: {suggested_max:.1f}")
                print(f"vix_check_enabled: true")
                print("-"*80)
        else:
            print("\n⚠️ NO CONSISTENTLY PROFITABLE RANGES YET")
            print("   Keep collecting data. Need more trades in each VIX range.")
            print("   Minimum 3 trades per range with win rate ≥50% required.\n")
        
        # Warning about ranges to avoid
        losing_ranges = []
        for vix_range, data in stats['by_vix_range'].items():
            if data['total_pnl'] < -500 or (data['win_rate'] < 40 and data['trades'] >= 3):
                losing_ranges.append((vix_range, data))
        
        if losing_ranges:
            print("\n✗ VIX RANGES TO AVOID (Poor performance):")
            for vix_range, data in losing_ranges:
                print(f"  • {vix_range}: {data['win_rate']}% win rate, ₹{data['total_pnl']} total P&L ({data['trades']} trades)")
        
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("1. Continue forward testing for at least 2-4 weeks")
        print("2. Aim for minimum 30-50 total trades across all VIX ranges")
        print("3. Re-run this analysis weekly to refine VIX thresholds")
        print("4. Once confident, update config.yaml with suggested settings")
        print("5. Enable vix_check_enabled: true to start filtering trades")
        print("="*80 + "\n")
    else:
        print(f"\n⚠️ {stats.get('error', 'Unknown error')}\n")
        if "total_trades" in stats:
            print(f"Total trades in history: {stats.get('total_trades', 0)}")
        if "trades_with_vix" in stats:
            print(f"Trades with VIX data: {stats['trades_with_vix']}")
        if "message" in stats:
            print(f"\n{stats['message']}")
        print("\nKeep trading to collect VIX data across different market conditions!")
        print()

if __name__ == "__main__":
    main()
