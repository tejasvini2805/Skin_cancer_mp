"""Generate synthetic JSONL dataset for OncoDetect LLM fine-tuning.

Usage:
  python generate_synthetic_dataset.py --count 500 --out llm_synthetic.jsonl
"""
import json
import random
import argparse
from datetime import datetime, timedelta

CLASS_NAMES = ['AKIEC','BCC','BKL','DF','MEL','NV','SCC','VASC']
LOCATIONS = ['upper extremity','lower extremity','anterior torso','posterior torso','head/neck','palms/soles','unknown']
GENDERS = ['male','female','unknown']

TEMPLATES = {
    'benign': "summary: The model predicts '{pred}' with high confidence ({conf}%).\nheatmap_insights: Single central hotspot consistent with uniform pigmentation.\nexplanation: Features suggest benign lesion.\nrecommendation: Low priority - self-monitor and photograph every 3 months.\nuncertainty: Low.",
    'suspicious': "summary: Model predicts '{pred}' with moderate confidence ({conf}%).\nheatmap_insights: Heatmap highlights irregular margins and asymmetric darker regions.\nexplanation: Visual features and patient metadata increase concern.\nrecommendation: Urgent dermatology referral for dermoscopy/biopsy.\nuncertainty: Moderate.",
    'low_conf': "summary: Prediction '{pred}' with low confidence ({conf}%).\nheatmap_insights: Diffuse low activation, no focal hotspots.\nexplanation: Model uncertain; metadata may conflict.\nrecommendation: Clinician triage recommended.\nuncertainty: High."
}


def random_confidence(kind):
    if kind=='benign': return round(random.uniform(80,98),1)
    if kind=='suspicious': return round(random.uniform(50,75),1)
    return round(random.uniform(30,55),1)


def gen_item(i):
    kind = random.choice(['benign','suspicious','low_conf'])
    pred = random.choice(CLASS_NAMES)
    conf = random_confidence(kind)
    hist = {
        'id': f'syn{i:05d}',
        'prediction': pred,
        'confidence': conf,
        'raw_class': pred,
        'timestamp': (datetime.utcnow() - timedelta(days=random.randint(0,30))).isoformat()+'Z',
        'age': str(random.randint(12,85)),
        'gender': random.choice(GENDERS),
        'location': random.choice(LOCATIONS),
        'confidence_breakdown': {c: round(random.uniform(0,100),1) for c in CLASS_NAMES}
    }
    # normalize breakdown so it sums to ~100
    total = sum(hist['confidence_breakdown'].values())
    for k in hist['confidence_breakdown']:
        hist['confidence_breakdown'][k] = round(hist['confidence_breakdown'][k]*100.0/total,1)

    heatmap_summary = ''
    if kind=='benign': heatmap_summary = 'Central uniform hotspot covering lesion core.'
    elif kind=='suspicious': heatmap_summary = 'Irregular peripheral hotspots and asymmetric dark region.'
    else: heatmap_summary = 'Diffuse low activation; no focal hotspots.'

    assistant_response = TEMPLATES[kind].format(pred=pred, conf=conf)

    return {'id': f'syn{i:05d}', 'history_item': hist, 'heatmap_summary': heatmap_summary, 'question': 'Please analyze', 'assistant_response': assistant_response}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=200, help='How many examples to generate')
    parser.add_argument('--out', type=str, default='llm_synthetic.jsonl')
    args = parser.parse_args()

    with open(args.out, 'w', encoding='utf-8') as f:
        for i in range(args.count):
            item = gen_item(i)
            f.write(json.dumps(item) + '\n')
    print(f'Wrote {args.count} synthetic examples to {args.out}')

if __name__ == '__main__':
    main()
