
import numpy as np
import pandas as pd

def generate_synthetic_interactions(num_students=50, sessions_per_student=30, seed=42):
    np.random.seed(seed)
    rows = []
    for sid in range(num_students):
        baseline = np.random.uniform(0.3, 0.8)  # baseline mastery
        for s in range(sessions_per_student):
            content_id = np.random.randint(0, 20)
            time_spent = max(5, int(np.random.normal(60, 20)))
            # simulate score influenced by baseline + small noise + content fit
            content_fit = np.random.choice([0,1], p=[0.3,0.7])  # whether content variant fits
            score = min(1.0, max(0.0, baseline + (0.2 if content_fit else -0.1) + np.random.normal(0,0.15)))
            rows.append({
                'student_id': f'st{sid:03d}',
                'session': s,
                'content_id': int(content_id),
                'time_spent': int(time_spent),
                'content_fit': int(content_fit),
                'score': float(round(score,3))
            })
    return pd.DataFrame(rows)

if __name__ == '__main__':
    df = generate_synthetic_interactions(20, 20)
    df.to_csv('synthetic_interactions.csv', index=False)
    print('Wrote synthetic_interactions.csv (rows=', len(df), ')')
