"""Reproduce the V17–V19 controlled ablations in fresh artifacts; no submission upload."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['rank48','aux4','shrink40','shrink80','shrink60','seed43','slow43','seed44','signed','signed025','decay','adaptive'], default='rank48')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--screen', type=Path, help='Reuse a completed day-26 factor run with matching configuration')
    parser.add_argument('--full', action='store_true', help='Refit and generate CSV only if the checks support the candidate')
    parser.add_argument('--broad', action='store_true', help='Also confirm the January 24 first-user frame')
    args = parser.parse_args()
    number, dim, aux, shrink = {'rank48':(17,48,2,20), 'aux4':(18,32,4,20), 'shrink40':(19,32,2,40), 'shrink80':(20,32,2,80), 'shrink60':(21,32,2,60), 'seed43':(22,32,2,40), 'slow43':(23,32,2,40), 'seed44':(24,32,2,40), 'signed':(25,32,2,40), 'signed025':(30,32,2,40), 'decay':(31,32,2,40), 'adaptive':(32,32,2,40)}[args.variant]
    version=f'v{number}'
    tag=f'{version}_{args.variant}'
    signed = args.variant in ('signed', 'signed025', 'decay', 'adaptive')
    decay = args.variant == 'decay'
    adaptive = args.variant == 'adaptive'
    negative_weight = .025 if args.variant in ('signed025', 'decay') else .1
    added_member = args.variant == 'seed44' or signed
    seed = {'seed43': 43, 'slow43': 43, 'seed44': 44}.get(args.variant, 42)
    ensemble = seed != 42 or signed
    slow = args.variant == 'slow43'
    epoch_budget = 24 if slow else 12
    learning_rate = .0012 if slow else .0024
    def model_name(day): return f'{tag}_d{dim}_day{day}_s{seed}'
    baseline = 'v19' if number >= 20 else 'v16'
    baseline_dir = ROOT/'arena_runs'
    baseline_tag = 'v16_confidence'
    if baseline == 'v19':
        baseline_dir /= 'v19_checks'
        baseline_tag = 'v19_shrink40'
    comparison_dir = ROOT/'arena_runs'/'v22_checks' if slow or added_member else baseline_dir
    comparison_version = 'v22' if slow or added_member else baseline
    if signed:
        comparison_dir = ROOT/'arena_runs'/'v24_checks'
        comparison_version = 'v24'
    if decay:
        comparison_dir = ROOT/'arena_runs'/'v25_checks'
        comparison_version = 'v25'
    if adaptive:
        comparison_dir = ROOT/'arena_runs'/'v31_verified'
        comparison_version = 'v31'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = args.out or ROOT/'arena_reproductions'/f'{version}_checks_{stamp}'
    run = run.resolve()
    run.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, ARENA_RUNS=str(run), OPENBLAS_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1')

    def execute(script, *parts):
        subprocess.run([sys.executable, str(ROOT/'arena'/script), *map(str, parts)],
                       cwd=ROOT, env=env, check=True)

    execute('test_context_v8.py')
    execute('test_time_history_v10.py')
    execute('test_cold_history_v14.py')
    execute('test_profile_shrinkage_v16.py')
    execute('prepare.py')
    common = ['--mode', 'baseline', '--gates', '--gate-mode', 'global',
              '--supervised-loss', 'pairwise', '--aux-controls', 'observed',
              '--context-rank', '4', '--context-mode', 'bucket', '--aux', str(aux), '--user-dropout', '.1', '--profile-shrinkage', str(shrink), '--dim', str(dim), '--lr', str(learning_rate)]
    if signed:
        common += ['--spectral-negative-weight', str(negative_weight)]

    if decay:
        common += ['--spectral-half-life-days', '7']
    if adaptive:
        common += ['--spectral-adaptive-negatives']
    member_weight = 1/6 if adaptive else .2 if decay else .25 if signed else 1/3 if added_member else .5

    def blend_member(candidate, candidate_events, reference, reference_events, out):
        if not ensemble:
            return candidate
        execute('blend_v3.py', '--raw', candidate, '--combined', reference,
                '--raw-events', candidate_events, '--combined-events', reference_events,
                '--raw-weight', member_weight, '--out', out)
        return out

    def raw_prediction(factor, day):
        ref = baseline_dir/f'{baseline_tag}_d32_day{day}_s42'
        return blend_member(factor/'valid_pred.npy', factor/'valid_events.npy',
                            reference_raw(day) if added_member else ref/'valid_pred.npy', ref/'valid_events.npy',
                            factor/'ensemble_valid.npy')

    def reference_raw(day):
        if adaptive:
            return comparison_dir/f'v31_decay_d32_day{day}_s42/ensemble_valid.npy'
        if decay:
            return comparison_dir/f'v25_signed_d32_day{day}_s42/ensemble_valid.npy'
        if signed:
            return comparison_dir/f'v24_seed44_d32_day{day}_s44/ensemble_valid.npy'
        if slow or added_member:
            return comparison_dir/f'v22_seed43_d32_day{day}_s43/ensemble_valid.npy'
        return baseline_dir/f'{baseline_tag}_d32_day{day}_s42/valid_pred.npy'

    for day in [26, 25]:
        cap = ['--selection-end-day', '26'] if day == 25 else []
        factor = run/model_name(day)
        if day==26 and args.screen:
            config=json.loads((args.screen/'config.json').read_text())
            expected=dict(day=26, tag=tag, dim=dim, aux=aux, profile_shrinkage=shrink,
                          seed=seed, epochs=epoch_budget, lr=learning_rate, full=False, selection_end_day=None,
                          context_mode='bucket', context_rank=4, mode='baseline',
                          gates=True, gate_mode='global', supervised_loss='pairwise',
                          aux_controls='observed', user_dropout=.1)
            if signed:
                expected['spectral_negative_weight'] = negative_weight
            if decay:
                expected['spectral_half_life_days'] = 7
            if adaptive:
                expected['spectral_adaptive_negatives'] = True
            if any(config.get(key)!=value for key,value in expected.items()):
                raise ValueError('Screen configuration does not match this variant')
            shutil.copytree(args.screen, factor)
        else:
            execute('history_model_v3.py', '--day', day, '--tag', tag,
                    '--epochs', epoch_budget, '--seed', seed, *common, *cap)
        execute('compare_v3.py', '--candidate', raw_prediction(factor, day),
                '--events', factor/'valid_events.npy', '--day', day,
                '--baseline', reference_raw(day),
                '--out', run/f'{version}_raw{day}_vs_{comparison_version}.json')
        result=json.loads((run/f'{version}_raw{day}_vs_{comparison_version}.json').read_text())
        if day==26 and result['global_auc_delta']<=0:
            print(f'Main raw comparison declined. Retain {comparison_version}. Artifacts:',run)
            return
        execute('history_model_v3.py', '--export', factor)
    early = model_name(25)
    later = model_name(26)
    if args.broad:
        execute('history_model_v3.py','--day','24','--selection-end-day','25',
                '--tag',tag,'--epochs',30 if slow else 15,'--seed',seed,*common)
        p=run/model_name(24)
        execute('compare_v3.py','--candidate',raw_prediction(p,24),'--events',p/'valid_events.npy',
                '--day','24','--baseline',reference_raw(24),
                '--out',run/f'{version}_raw24_vs_{comparison_version}.json')
    execute('calibrate.py', '--train-model', early, '--test-model', later, '--out', f'{version}_cal_confidence')
    cal = run/f'{version}_cal_confidence'
    factor = run/later
    execute('blend_v3.py', '--raw', factor/'valid_pred.npy', '--combined', cal/'valid_pred.npy',
            '--raw-events', factor/'valid_events.npy', '--combined-events', cal/'valid_events.npy',
            '--out', cal/'valid_blend.npy')
    rounds = json.loads((cal/'config.json').read_text())['rounds']
    execute('event_component_v3.py', '--train-model', early, '--test-model', later,
            '--row-rounds', rounds, '--row-pred', cal/'valid_blend.npy',
            '--row-events', factor/'valid_events.npy', '--out', f'{version}_event_confidence')
    event = run/f'{version}_event_confidence'
    ref_event = baseline_dir/f'{baseline}_event_confidence'
    reference_final = comparison_dir/f'{comparison_version}_event_confidence/ensemble_valid.npy' if slow or added_member else ref_event/'valid_half_corrected.npy'
    valid_final = blend_member(event/'valid_half_corrected.npy', event/'valid_events.npy',
                               reference_final if added_member else ref_event/'valid_half_corrected.npy', ref_event/'valid_events.npy',
                               event/'ensemble_valid.npy')
    execute('compare_v3.py', '--candidate', valid_final,
            '--events', event/'valid_events.npy',
            '--baseline', reference_final,
            '--bootstrap', '150', '--out', run/f'{version}_vs_{comparison_version}.json')
    print(f'{version} comparison artifacts:', run)
    comparison=json.loads((run/f'{version}_vs_{comparison_version}.json').read_text())
    earlier=json.loads((run/f'{version}_raw25_vs_{comparison_version}.json').read_text())
    supported=(comparison['bootstrap']['paired_delta_95_percentile_interval'][0]>0
               and earlier['global_auc_delta']>0)
    if args.broad:
        supported=supported and json.loads((run/f'{version}_raw24_vs_{comparison_version}.json').read_text())['global_auc_delta']>0
    if not args.full or not supported:
        print(f'No full refit requested or insufficient evidence. {comparison_version} preserved.')
        return
    epochs=json.loads((factor/'best.json').read_text())['epoch']
    execute('history_model_v3.py','--day','26','--tag',f'{tag}_final','--epochs',epochs,'--seed',seed,'--full',*common)
    full=run/f'{tag}_final_d{dim}_day26_s{seed}_full'
    with np.load(full/'calibration_features.npz') as c:np.save(full/'evaluation_events.npy',c['events'])
    execute('calibrate_v3.py','--apply-model',cal/'model.txt','--test-model',full.name,
            '--out',f'{version}_final_combiner','--fulltest')
    final=run/f'{version}_final_combiner'
    execute('blend_v3.py','--raw',full/'evaluation_pred.npy','--combined',final/'evaluation_pred.npy',
            '--raw-events',full/'evaluation_events.npy','--combined-events',final/'events.npy',
            '--out',final/'evaluation_blend.npy')
    execute('v3_event_apply.py','--features',full/'calibration_features.npz','--rows',final/'evaluation_blend.npy',
            '--row-events',final/'events.npy','--event-model',event/'event_model.txt','--out',final/'evaluation_final.npy')
    ref_final = comparison_dir/f'{comparison_version}_final_combiner' if added_member else baseline_dir/f'{baseline}_final_combiner'
    evaluation_final = blend_member(final/'evaluation_final.npy', final/'events.npy',
                                    ref_final/('ensemble_evaluation.npy' if added_member else 'evaluation_final.npy'), ref_final/'events.npy',
                                    final/'ensemble_evaluation.npy')
    execute('submission.py','--predictions',evaluation_final,
            '--out',f'submission_techarena_{version}_reproduced_{stamp}.csv')


if __name__ == '__main__':
    main()
