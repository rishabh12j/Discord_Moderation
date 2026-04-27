import os
import json
import numpy as np
from collections import Counter
from typing import Dict, Any


def assess_thread_quality(
    threads_file: str = "data/processed/multilingual_threads.json",
    output_file: str = "data/logs/thread_quality_report.json"
) -> Dict[str, Any]:
    """
    Compute diversity metrics for synthetic threads to detect LLM repetition issues.
    """
    if not os.path.exists(threads_file):
        raise FileNotFoundError(f"Input file {threads_file} not found.")
    
    print(f"Loading threads from {threads_file}...")
    with open(threads_file, 'r', encoding='utf-8') as f:
        threads = json.load(f)
    
    # Basic counts
    total_threads = len(threads)
    all_messages = [msg for thread in threads for msg in thread.get('messages', [])]
    total_messages = len(all_messages)
    
    # Extract all texts and user IDs
    all_texts = [msg.get('text', '').strip() for msg in all_messages if msg.get('text')]
    all_user_ids = [msg.get('user_id', '') for msg in all_messages if msg.get('user_id')]
    
    # Language distribution
    language_dist = Counter(thread.get('language', 'unknown') for thread in threads)
    
    # Messages per thread
    messages_per_thread = [len(thread.get('messages', [])) for thread in threads]
    
    # Text diversity (key quality indicator)
    unique_texts = set(all_texts)
    text_diversity = len(unique_texts) / len(all_texts) if all_texts else 0
    
    # User diversity
    unique_users = set(all_user_ids)
    
    # Toxicity distribution (if available)
    toxic_counts = Counter()
    for thread in threads:
        for msg in thread.get('messages', []):
            if isinstance(msg, dict) and 'toxic' in msg:
                toxic_counts[msg['toxic']] += 1
    
    # Find most repeated texts (sign of LLM repetition)
    text_counts = Counter(all_texts)
    most_common_texts = text_counts.most_common(10)
    
    # Check thread length distribution
    short_threads = sum(1 for length in messages_per_thread if length < 3)
    medium_threads = sum(1 for length in messages_per_thread if 3 <= length <= 6)
    long_threads = sum(1 for length in messages_per_thread if length > 6)
    
    metrics = {
        "summary": {
            "total_threads": total_threads,
            "total_messages": total_messages,
            "unique_messages": len(unique_texts),
            "unique_users": len(unique_users),
            "text_diversity_ratio": round(text_diversity, 4)
        },
        "thread_statistics": {
            "avg_messages_per_thread": round(np.mean(messages_per_thread), 2),
            "median_messages_per_thread": int(np.median(messages_per_thread)),
            "min_messages": int(np.min(messages_per_thread)),
            "max_messages": int(np.max(messages_per_thread)),
            "short_threads_(<3_msgs)": short_threads,
            "medium_threads_(3-6_msgs)": medium_threads,
            "long_threads_(>6_msgs)": long_threads
        },
        "language_distribution": dict(language_dist),
        "quality_indicators": {
            "text_diversity": {
                "score": round(text_diversity, 4),
                "interpretation": (
                    "Excellent (>0.9)" if text_diversity > 0.9 else
                    "Good (0.7-0.9)" if text_diversity > 0.7 else
                    "Acceptable (0.5-0.7)" if text_diversity > 0.5 else
                    "Poor (<0.5) - LLM may be repeating"
                )
            },
            "user_diversity": {
                "unique_users": len(unique_users),
                "avg_users_per_thread": round(len(unique_users) / total_threads, 2)
            }
        },
        "repetition_analysis": {
            "top_10_repeated_texts": [
                {
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "count": count,
                    "percentage": round((count / total_messages) * 100, 2)
                }
                for text, count in most_common_texts
            ]
        }
    }
    
    # Save metrics
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 Thread Quality Assessment")
    print(f"{'='*60}")
    print(f"\n📈 Summary:")
    print(f"   Total Threads: {total_threads}")
    print(f"   Total Messages: {total_messages}")
    print(f"   Unique Messages: {len(unique_texts)}")
    print(f"   Text Diversity: {text_diversity:.4f} ({metrics['quality_indicators']['text_diversity']['interpretation']})")
    
    print(f"\n📏 Thread Length Distribution:")
    print(f"   Short (<3 msgs): {short_threads}")
    print(f"   Medium (3-6 msgs): {medium_threads}")
    print(f"   Long (>6 msgs): {long_threads}")
    
    print(f"\n🌍 Language Distribution:")
    for lang, count in sorted(language_dist.items(), key=lambda x: -x[1])[:10]:
        print(f"   {lang}: {count} threads")
    
    print(f"\n⚠️  Most Repeated Texts:")
    for i, (text, count) in enumerate(most_common_texts[:5], 1):
        preview = text[:80] + "..." if len(text) > 80 else text
        print(f"   {i}. [{count}x] {preview}")
    
    print(f"\n✅ Full report saved to: {output_file}")
    
    return metrics


if __name__ == "__main__":
    assess_thread_quality()
