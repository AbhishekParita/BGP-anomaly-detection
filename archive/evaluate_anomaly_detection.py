"""
Evaluate Ensemble Model's Detection of 14 Anomaly Types
This script compares the ensemble model's detections with the actual anomalies in the dataset.
"""

import pandas as pd
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ensemble_bgp_optimized import run_anomaly_scoring

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# --- CONFIGURATION ---
NEW_DATA_PATH = 'datasetfiles/bgp_test_data_60d_clean.csv'  # Update if needed
OUTPUT_DIR = 'anomaly_detection_evaluation'

def evaluate_anomaly_detection():
    """Evaluate which anomalies are being detected by the ensemble model."""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 80)
    print("ENSEMBLE MODEL - ANOMALY TYPE DETECTION EVALUATION")
    print("=" * 80)
    
    # 1. Load dataset with ground truth anomaly types
    print(f"\n📂 Loading data: {NEW_DATA_PATH}")
    try:
        df_raw = pd.read_csv(NEW_DATA_PATH)
        df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
        print(f"✅ Loaded {len(df_raw)} records")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # 2. Run ensemble scoring
    print("\n🔬 Running ensemble anomaly scoring...")
    results_df, thresholds = run_anomaly_scoring(df_raw)
    
    if results_df.empty:
        print("❌ No results generated")
        return
    
    print(f"✅ Scored {len(results_df)} records")
    
    # 3. Get ground truth anomaly types if available
    print("\n📊 Analyzing anomaly types in dataset...")
    
    if 'anomaly_type' in df_raw.columns:
        # Map results back to original data by timestamp + peer
        df_raw['has_timestamp'] = df_raw.index
        df_merged = results_df.copy()
        
        # Extract ground truth anomaly types (note: results_df has fewer rows due to sequence creation)
        # We'll align based on timestamps
        
        # Count ground truth anomalies
        ground_truth_anomalies = df_raw[df_raw['anomaly_type'] != 'normal']['anomaly_type'].value_counts()
        
        print("\n📍 GROUND TRUTH - Anomalies in Dataset:")
        print("-" * 60)
        for anom_type, count in ground_truth_anomalies.items():
            pct = (count / len(df_raw)) * 100
            print(f"  {anom_type:25s}: {count:5d} records ({pct:5.1f}%)")
        
        total_anomalies_gt = ground_truth_anomalies.sum()
        print(f"  {'TOTAL ANOMALIES':25s}: {total_anomalies_gt:5d} records ({(total_anomalies_gt/len(df_raw))*100:5.1f}%)")
    
    # 4. Analyze model detections
    print("\n🎯 MODEL DETECTIONS - Anomalies Found by Ensemble:")
    print("-" * 60)
    
    detected_anomalies = results_df[results_df['severity'] != 'NORMAL']
    
    severity_counts = detected_anomalies['severity'].value_counts()
    for severity, count in severity_counts.items():
        pct = (count / len(results_df)) * 100
        print(f"  {severity:25s}: {count:5d} records ({pct:5.1f}%)")
    
    print(f"  {'TOTAL DETECTED':25s}: {len(detected_anomalies):5d} records ({(len(detected_anomalies)/len(results_df))*100:5.1f}%)")
    
    # 5. Detection metrics
    print("\n📈 DETECTION METRICS:")
    print("-" * 60)
    
    if 'anomaly_type' in df_raw.columns:
        total_gt_anomalies = len(df_raw[df_raw['anomaly_type'] != 'normal'])
        total_detected = len(detected_anomalies)
        
        # Simple detection rate (what % of all records with anomalies got flagged)
        # This is approximate since results_df has different row count
        detection_rate = (total_detected / total_gt_anomalies * 100) if total_gt_anomalies > 0 else 0
        
        print(f"  Total Ground Truth Anomalies    : {total_gt_anomalies}")
        print(f"  Total Detected by Model         : {total_detected}")
        print(f"  Approximate Detection Rate      : {detection_rate:.1f}%")
        print(f"  Average Confidence (Detected)   : {detected_anomalies['confidence'].mean():.1f}%")
        print(f"  Model Agreement (Both Models)   : {detected_anomalies['both_agree'].sum()} records")
    
    # 6. Severity distribution analysis
    print("\n🔴 SEVERITY DISTRIBUTION:")
    print("-" * 60)
    
    severity_dist = results_df['severity'].value_counts()
    for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NORMAL']:
        if severity in severity_dist.index:
            count = severity_dist[severity]
            pct = (count / len(results_df)) * 100
            print(f"  {severity:25s}: {count:5d} records ({pct:5.1f}%)")
    
    # 7. Save detailed results
    print("\n💾 Saving results...")
    
    # Save full results
    results_path = os.path.join(OUTPUT_DIR, 'anomaly_detection_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"  ✅ Full results: {results_path}")
    
    # Save alerts only
    alerts_path = os.path.join(OUTPUT_DIR, 'detected_alerts.csv')
    detected_anomalies.to_csv(alerts_path, index=False)
    print(f"  ✅ Alerts only: {alerts_path}")
    
    # Save evaluation summary
    summary = {
        "evaluation_date": pd.Timestamp.now().isoformat(),
        "total_records_scored": len(results_df),
        "total_anomalies_detected": len(detected_anomalies),
        "detection_percentage": (len(detected_anomalies) / len(results_df) * 100),
        "severity_breakdown": severity_dist.to_dict(),
        "average_confidence": float(detected_anomalies['confidence'].mean()) if len(detected_anomalies) > 0 else 0,
        "model_agreement_count": int(detected_anomalies['both_agree'].sum()),
        "thresholds": {k: float(v) for k, v in thresholds.items()}
    }
    
    if 'anomaly_type' in df_raw.columns:
        summary["ground_truth_anomalies"] = ground_truth_anomalies.to_dict()
        summary["approximate_detection_rate"] = detection_rate
    
    summary_path = os.path.join(OUTPUT_DIR, 'evaluation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✅ Summary: {summary_path}")
    
    # 8. Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS:")
    print("=" * 80)
    
    if len(detected_anomalies) > 0:
        print(f"✅ Model IS DETECTING ANOMALIES")
        print(f"   - Detected: {len(detected_anomalies)} anomalies")
        print(f"   - Detection rate: {(len(detected_anomalies)/len(results_df)*100):.1f}%")
        print(f"   - Average confidence: {detected_anomalies['confidence'].mean():.1f}%")
        
        high_severity = len(detected_anomalies[detected_anomalies['severity'].isin(['HIGH', 'CRITICAL'])])
        if high_severity > 0:
            print(f"   - High/Critical alerts: {high_severity} ({(high_severity/len(detected_anomalies)*100):.1f}%)")
    else:
        print("⚠️  Model did NOT detect any anomalies")
    
    print("\n⚠️  LIMITATION:")
    print("   The ensemble model detects THAT an anomaly exists, but does NOT identify")
    print("   WHICH TYPE of anomaly (e.g., Route Hijack vs Bogon ASN vs RPKI Invalid).")
    print("   Currently, it only classifies into severity levels (NORMAL/LOW/MEDIUM/HIGH/CRITICAL).")
    
    print("\n" + "=" * 80)
    
    # 9. Create visualizations
    print("\n📊 Creating visualizations...")
    create_visualizations(df_raw, results_df, detected_anomalies, OUTPUT_DIR)
    print("✅ Visualizations saved to anomaly_detection_evaluation/")


def create_visualizations(df_raw, results_df, detected_anomalies, output_dir):
    """Create comprehensive visualizations of anomaly detection results."""
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 14))
    
    # 1. GROUND TRUTH ANOMALY DISTRIBUTION
    if 'anomaly_type' in df_raw.columns:
        ax1 = plt.subplot(3, 3, 1)
        gt_counts = df_raw[df_raw['anomaly_type'] != 'normal']['anomaly_type'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(gt_counts)))
        gt_counts.plot(kind='barh', ax=ax1, color=colors)
        ax1.set_title('Ground Truth: Anomaly Type Distribution', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Count')
        for i, v in enumerate(gt_counts.values):
            ax1.text(v + 0.5, i, str(v), va='center')
    
    # 2. SEVERITY DISTRIBUTION (PIE)
    ax2 = plt.subplot(3, 3, 2)
    severity_counts = results_df['severity'].value_counts()
    colors_severity = {'CRITICAL': '#d62728', 'HIGH': '#ff7f0e', 'MEDIUM': '#ffbb78', 
                       'LOW': '#98df8a', 'NORMAL': '#2ca02c'}
    colors_list = [colors_severity.get(s, '#1f77b4') for s in severity_counts.index]
    ax2.pie(severity_counts.values, labels=severity_counts.index, autopct='%1.1f%%', 
            colors=colors_list, startangle=90)
    ax2.set_title('Model Output: Severity Distribution', fontsize=12, fontweight='bold')
    
    # 3. DETECTED VS UNDETECTED
    ax3 = plt.subplot(3, 3, 3)
    detected_count = len(detected_anomalies)
    normal_count = len(results_df[results_df['severity'] == 'NORMAL'])
    detection_data = [normal_count, detected_count]
    colors_det = ['#2ca02c', '#d62728']
    ax3.pie(detection_data, labels=['Normal', 'Anomalies'], autopct='%1.1f%%', 
            colors=colors_det, startangle=90, textprops={'fontsize': 11})
    ax3.set_title(f'Detection Summary\n({detected_count} anomalies detected)', 
                  fontsize=12, fontweight='bold')
    
    # 4. CONFIDENCE SCORE DISTRIBUTION
    ax4 = plt.subplot(3, 3, 4)
    ax4.hist(results_df['confidence'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax4.axvline(results_df['confidence'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {results_df["confidence"].mean():.1f}%')
    ax4.set_xlabel('Confidence Score (%)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Confidence Score Distribution (All Records)', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # 5. CONFIDENCE SCORE - DETECTED ONLY
    ax5 = plt.subplot(3, 3, 5)
    if len(detected_anomalies) > 0:
        ax5.hist(detected_anomalies['confidence'], bins=30, color='coral', 
                 edgecolor='black', alpha=0.7)
        ax5.axvline(detected_anomalies['confidence'].mean(), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {detected_anomalies["confidence"].mean():.1f}%')
        ax5.set_xlabel('Confidence Score (%)')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Confidence Score Distribution (Detected Anomalies Only)', 
                      fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(alpha=0.3)
    
    # 6. LSTM ERROR vs IF SCORE (SCATTER)
    ax6 = plt.subplot(3, 3, 6)
    scatter_colors = ['red' if s != 'NORMAL' else 'green' for s in results_df['severity']]
    ax6.scatter(results_df['lstm_error'], results_df['if_score'], 
               c=scatter_colors, alpha=0.5, s=30)
    ax6.set_xlabel('LSTM Reconstruction Error')
    ax6.set_ylabel('Isolation Forest Score')
    ax6.set_title('Model Components Scatter Plot\n(Red=Anomaly, Green=Normal)', 
                  fontsize=12, fontweight='bold')
    ax6.grid(alpha=0.3)
    
    # 7. SEVERITY BY COUNT
    ax7 = plt.subplot(3, 3, 7)
    severity_order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NORMAL']
    severity_data = results_df['severity'].value_counts().reindex(severity_order, fill_value=0)
    colors_sev = ['#d62728', '#ff7f0e', '#ffbb78', '#98df8a', '#2ca02c']
    bars = ax7.bar(severity_order, severity_data.values, color=colors_sev, edgecolor='black', linewidth=1.5)
    ax7.set_ylabel('Count')
    ax7.set_title('Severity Level Distribution', fontsize=12, fontweight='bold')
    ax7.grid(axis='y', alpha=0.3)
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # 8. MODEL AGREEMENT
    ax8 = plt.subplot(3, 3, 8)
    agreement_counts = results_df['both_agree'].value_counts()
    agreement_labels = ['Disagreement', 'Agreement']
    agreement_data = [
        agreement_counts.get(0, 0),
        agreement_counts.get(1, 0)
    ]
    colors_agree = ['#ff7f0e', '#2ca02c']
    ax8.bar(agreement_labels, agreement_data, color=colors_agree, edgecolor='black', linewidth=1.5)
    ax8.set_ylabel('Count')
    ax8.set_title('LSTM & IF Model Agreement', fontsize=12, fontweight='bold')
    ax8.grid(axis='y', alpha=0.3)
    for i, v in enumerate(agreement_data):
        ax8.text(i, v + 10, str(v), ha='center', va='bottom', fontweight='bold')
    
    # 9. ENSEMBLE SCORE DISTRIBUTION
    ax9 = plt.subplot(3, 3, 9)
    ax9.hist(results_df['ensemble_score'], bins=50, color='mediumpurple', 
             edgecolor='black', alpha=0.7)
    ax9.axvline(results_df['ensemble_score'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {results_df["ensemble_score"].mean():.2f}')
    ax9.set_xlabel('Ensemble Score')
    ax9.set_ylabel('Frequency')
    ax9.set_title('Ensemble Score Distribution', fontsize=12, fontweight='bold')
    ax9.legend()
    ax9.grid(alpha=0.3)
    
    plt.tight_layout()
    viz_path = os.path.join(output_dir, 'anomaly_detection_overview.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {viz_path}")
    plt.close()
    
    # ADDITIONAL FIGURE: Time-based analysis
    if 'timestamp' in results_df.columns:
        fig2, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # 1. Detections over time
        ax = axes[0, 0]
        results_by_time = results_df.groupby('timestamp').apply(
            lambda x: (x['severity'] != 'NORMAL').sum()
        )
        ax.plot(results_by_time.index, results_by_time.values, color='red', linewidth=2, marker='o')
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Anomalies Detected')
        ax.set_title('Anomalies Detected Over Time', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. Average confidence over time
        ax = axes[0, 1]
        results_by_time_conf = results_df.groupby('timestamp')['confidence'].mean()
        ax.plot(results_by_time_conf.index, results_by_time_conf.values, 
               color='blue', linewidth=2, marker='o')
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Average Confidence (%)')
        ax.set_title('Average Confidence Score Over Time', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 3. Ensemble score over time
        ax = axes[1, 0]
        results_by_time_score = results_df.groupby('timestamp')['ensemble_score'].mean()
        ax.plot(results_by_time_score.index, results_by_time_score.values, 
               color='purple', linewidth=2, marker='o')
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Average Ensemble Score')
        ax.set_title('Average Ensemble Score Over Time', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 4. Peer distribution (top 10)
        ax = axes[1, 1]
        if 'peer_addr' in results_df.columns:
            top_peers = detected_anomalies['peer_addr'].value_counts().head(10)
            top_peers.plot(kind='barh', ax=ax, color='skyblue', edgecolor='black')
            ax.set_xlabel('Anomalies Detected')
            ax.set_title('Top 10 Peers with Most Anomalies Detected', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        viz_time_path = os.path.join(output_dir, 'anomaly_detection_timeline.png')
        plt.savefig(viz_time_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved: {viz_time_path}")
        plt.close()


if __name__ == "__main__":
    evaluate_anomaly_detection()
