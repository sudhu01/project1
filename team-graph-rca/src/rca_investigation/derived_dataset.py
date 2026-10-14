"""Immutable encoding and an unrestricted validation diagnostic reference."""
from pathlib import Path
from dataclasses import replace
import json
from .contracts import digest
from .contracts_v2 import IncidentPublicV2
from .encoding import Encoder, derive_bundle
from .raw_dataset import load_raw_split
from .backends.evidence_v2 import RawEvidenceBackend
from .catalogue import catalogue_hash
from .dataset import atomic_json, verify_bundle, read_split, load_public_bundle, load_private_label
from .experiment import load_model, label_name


def encode_dataset(config, topology, specs):
    values, paths = config.values, config.values['paths']
    encoder = Encoder.load(paths['encoder'])
    if values['encoding_hash'] != encoder.hash:
        raise ValueError('task encoder not resolved')
    registry_payload = __import__('yaml').safe_load(Path(paths['resource_registry']).read_text())
    registry = registry_payload['resources']
    target = Path(paths['dataset'])
    if (target / 'splits.json').exists():
        raise ValueError('derived dataset already exists; declare a new version')
    raw_manifest, _ = load_raw_split(paths['raw_dataset'], 'train')
    if encoder.values['split_hash'] != raw_manifest['hash']:
        raise ValueError('encoder fitted on a different raw split')
    from .train import runtime_hash
    identity = {'raw_split_hash': raw_manifest['hash'], 'encoder_hash': encoder.hash, 'catalogue_hash': catalogue_hash(specs),
                'config_hash': config.hash, 'runtime_hash': runtime_hash()}
    identity_path = target / 'derivation.json'
    if target.exists() and not identity_path.exists():
        raise ValueError('existing derived directory has no frozen derivation identity')
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError('partial derivation inputs or source changed')
    else:
        atomic_json(identity_path, identity)
    hashes = {}
    for partition in ('train', 'validation', 'test'):
        _, rows = load_raw_split(paths['raw_dataset'], partition)
        for identifier, payloads in rows:
            if (target / identifier).exists():
                receipt = json.loads((target / identifier / 'capture_receipt.json').read_text())
                if receipt['encoder_hash'] != encoder.hash or receipt['catalogue_hash'] != catalogue_hash(specs) or receipt['raw_hash'] != raw_manifest['bundle_hashes'][identifier]:
                    raise ValueError('partial bundle encoding mismatch')
                hashes[identifier] = verify_bundle(target / identifier)
                continue
            raw_public = payloads['public.json']
            public = IncidentPublicV2(identifier, topology.hash, catalogue_hash(specs), 'front-end', 'manual_frontend',
                raw_public['window_start'], raw_public['window_end'], values['episode']['budget'], values['episode']['max_probes'],
                values['objective']['lambda_cost'], task_id=values['task_id'], encoding_hash=encoder.hash,
                objective_hash=digest(values['objective']))
            def project(spec, raw, public, encoder):
                if digest(raw['registry']) != raw_public['registry_hash']:
                    raise ValueError('raw registry identity mismatch')
                backend = RawEvidenceBackend(specs, raw, raw['registry'], topology, encoder)
                backend.begin(public)
                packet = backend.execute(spec)
                return packet
            path = derive_bundle(target, Path(paths['raw_dataset']) / identifier, public, topology, specs, encoder, project)
            hashes[identifier] = verify_bundle(path)
    derived = {key: value for key, value in raw_manifest.items() if key not in {'hash', 'bundle_hashes'}}
    derived.update(bundle_hashes=hashes, raw_split_hash=raw_manifest['hash'], encoder_hash=encoder.hash,
                   catalogue_hash=catalogue_hash(specs), config_hash=config.hash)
    derived['hash'] = digest(derived)
    atomic_json(target / 'splits.json', derived)
    return {'derived_incidents': len(hashes), 'derived_split_hash': derived['hash']}


def full_evidence_reference(config, specs):
    paths = config.values['paths']
    training, _ = read_split(paths['dataset'], 'train')
    _, validation = read_split(paths['dataset'], 'validation')
    model = load_model(paths['model'], config, specs, training['hash'])
    from collections import Counter
    grid, outcomes = {}, []
    for smoothing in (.05, .10, .20):
        candidate = replace(model, smoothing=smoothing)
        rows = []
        for path in validation:
            public, _, packets = load_public_bundle(path, specs)
            probabilities = candidate.diagnosis(tuple((packet.action_id, packet.symbol) for packet in packets))
            predicted = max(model.classes, key=lambda c: probabilities[c])
            truth = label_name(load_private_label(path))
            rows.append({'incident_id': public.incident_id, 'truth': truth, 'prediction': predicted,
                         'correct': predicted == truth, 'confidence': probabilities[predicted],
                         'brier': sum((p - (cause == truth))**2 for cause, p in probabilities.items())})
        grid[str(smoothing)] = {'accuracy': sum(row['correct'] for row in rows) / len(rows),
                              'brier': sum(row['brier'] for row in rows) / len(rows), 'model_hash': candidate.hash}
        if smoothing == model.smoothing:
            outcomes = rows
    if not outcomes:
        raise ValueError('configured smoothing lies outside the declared validation grid')
    accuracy = sum(row['correct'] for row in outcomes) / len(outcomes)
    return {'diagnostic_upper_reference_only': True, 'limited_budget_policy': False,
            'config_hash': config.hash, 'model_hash': model.hash, 'split_hash': training['hash'],
            'accuracy': accuracy, 'outcomes': outcomes, 'smoothing_grid': grid,
            'configured_smoothing': model.smoothing, 'smoothing_selection_applied': False,
            'training_promotion_passed': accuracy >= .9,
            'confusion': {cause: dict(Counter(row['prediction'] for row in outcomes if row['truth'] == cause)) for cause in model.classes},
            'scope': 'resource_pressure_included' if 'resource/carts' in model.classes else 'verified_checkout_incidents'}
