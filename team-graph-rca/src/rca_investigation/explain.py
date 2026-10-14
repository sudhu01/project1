"""Evidence-only explanation; provider integration is optional."""
import json


def explain_result(result):
    decision=result['decision']
    packets={p['action_id']:p for p in result['packets']}
    lines=[]
    for action,packet in packets.items():
        lines.append(f"{packet['resource_id']}: {packet['response_code']} via {action}.")
    prediction=decision['prediction']
    summary='Deferred because the acquired evidence does not support a diagnosis.' if prediction=='DEFER' else \
        f"Diagnosis: {prediction}. Model confidence: {decision['confidence']:.1%}."
    v2 = any(packet.get('schema_version') == 2 for packet in packets.values()) or any(
        cause.startswith(('resource/', 'edge/')) for cause in result.get('probabilities', {}))
    if not v2 and prediction not in ('DEFER','NO_FAULT'):
        summary+=' A stopped dependency can cause errors in a running caller.'
    explanation = {'summary':summary,'evidence':lines,'evidence_ids':decision['evidence_ids'],
            'stop_reason':decision['stop_reason'],'credits':result['credits'],
            'scope':'Catalogue database stop, orders stop, or healthy control only.'}
    if v2:
        explanation['scope'] = 'Registered causes: ' + ', '.join(result['probabilities']) + '.'
        explanation['mechanism_prediction_enabled'] = False
        explanation['log_excerpts'] = [{'action_id': action, 'lines': json.loads(packet['evidence_json'])['lines']}
            for action, packet in packets.items() if action.startswith('logs/') and packet['availability'] == 'OK']
        if 'resource/carts' in result['probabilities']:
            explanation['scope'] += ' Carts means retained resource pressure; customer degradation is not established.'
    return explanation
